"""
Multi-agent self-play + league training
─────────────────────────────────────────
Training curriculum:
  Phase 1 — Pure self-play (all 6 seats learn with shared weights)
  Phase 2 — League play (frozen checkpoints for diversity)
  Phase 3 — Fine-tune vs human-mimic opponents (scripted archetypes)

All seats share one AlphaPokerNet. Diversity comes from different
opponent history buffers. One checkpoint to download and deploy.
"""

import copy
import time
import random
import numpy as np
from pathlib import Path
from collections import deque

from .poker_env import PokerEnv, Action, NUM_ACTIONS
from .ppo_agent import PPOAgent, RolloutBuffer, AlphaPokerNet
from .opponent_encoder import InHandRecorder, EVENT_DIM, MAX_SEQ_LEN, LATENT_DIM
from .renderer import render_training_stats


# ── Hyper-parameters ──────────────────────────────────────────────────────────
CFG = {
    # Environment
    "num_players":      6,
    "starting_stack":   200.0,
    "big_blind":        2.0,
    "render_every":     5000,    # render table every N hands

    # Rollout
    "rollout_steps":    4096,    # steps per PPO update
    "batch_size":       512,
    "ppo_epochs":       4,
    "gamma":            0.99,
    "gae_lambda":       0.95,

    # Optimization
    "lr":               2.5e-4,
    "clip_eps":         0.2,
    "entropy_coef":     0.01,
    "value_coef":       0.5,

    # Exploration noise for non-hero seats
    "opp_epsilon":      0.15,    # probability of random action for opponents
    "opp_epsilon_decay": 0.9999, # decay per hand (0.15 → ~0.05 over 7K hands)

    # Self-play
    "self_play_update_freq":  500,
    "league_size":            8,
    "league_add_freq":        2000,
    "eval_freq":              1000,
    "eval_hands":             500,
    "min_elo_gain":           10.0,
    "log_freq":               500,      # print stats every N hands

    # Training phases
    "phase1_hands":     50_000,
    "phase2_hands":     200_000,
    "phase3_hands":     50_000,

    # Checkpointing
    "save_dir":         "checkpoints",
    "save_freq":        5000,
}


# ── ELO utilities ─────────────────────────────────────────────────────────────
def elo_update(winner_elo: float, loser_elo: float, k: float = 32.0) -> tuple:
    expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    delta    = k * (1 - expected)
    return winner_elo + delta, loser_elo - delta


# ── Scripted archetypes (Phase 3) ─────────────────────────────────────────────
class ScriptedAgent:
    """
    Simple rule-based agents that mimic common human player types.
    Used in Phase 3 to make the RL agent robust to real humans.
    """
    def __init__(self, style: str = "tight-aggressive"):
        self.style = style

    def get_action(self, obs: np.ndarray, legal_mask: np.ndarray,
                   deterministic: bool = False, **kwargs) -> tuple:
        """Returns (action, raise_frac, log_prob, value, entropy)."""
        legal = [i for i in range(NUM_ACTIONS) if legal_mask[i]]
        if not legal:
            return 0, 0.0, 0.0, 0.0, 0.0

        call_norm   = float(obs[129])
        street_oh   = obs[132:136]
        street_idx  = int(np.argmax(street_oh)) if any(street_oh) else 0
        call_high   = call_norm > 0.05

        raise_frac = 0.3

        if self.style == "nit":
            if call_high:
                action = Action.FOLD
            else:
                action = Action.CHECK

        elif self.style == "fish":
            action = Action.CALL if Action.CALL in legal else (
                Action.CHECK if Action.CHECK in legal else Action.FOLD)

        elif self.style == "maniac":
            if Action.ALL_IN in legal and random.random() < 0.3:
                action = Action.ALL_IN
            elif Action.RAISE in legal:
                action = Action.RAISE
                raise_frac = random.uniform(0.5, 1.0)
            else:
                action = random.choice(legal)

        elif self.style == "tight-aggressive":
            if street_idx == 0:
                if call_high:
                    if Action.RAISE in legal and random.random() < 0.5:
                        action = Action.RAISE
                        raise_frac = 0.25
                    else:
                        action = Action.CALL
                else:
                    if Action.RAISE in legal:
                        action = Action.RAISE
                        raise_frac = 0.20
                    else:
                        action = Action.CHECK
            else:
                action = random.choice(legal)
                if action == Action.RAISE:
                    raise_frac = random.uniform(0.15, 0.40)
        else:
            action = random.choice(legal)

        return int(action), raise_frac, 0.0, 0.0, 0.0


# ── League manager ────────────────────────────────────────────────────────────
class League:
    def __init__(self, max_size: int = 8):
        self.max_size  = max_size
        self.members:  list = []
        self.elo_hist: list = []

    def add(self, agent: PPOAgent, tag: str = ""):
        frozen = copy.deepcopy(agent)
        frozen.net.eval()
        self.members.append({
            "agent": frozen,
            "elo":   agent.elo,
            "tag":   tag,
            "added_at_elo": agent.elo,
        })
        if len(self.members) > self.max_size:
            self.members.sort(key=lambda m: m["elo"])
            self.members.pop(0)
        print(f"  League updated: {len(self.members)} members  "
              f"ELO range [{min(m['elo'] for m in self.members):.0f}"
              f"\u2013{max(m['elo'] for m in self.members):.0f}]")

    def sample(self) -> PPOAgent:
        if not self.members:
            return None
        return random.choice(self.members)["agent"]

    def sample_weighted(self) -> PPOAgent:
        if not self.members:
            return None
        elos   = np.array([m["elo"] for m in self.members], dtype=float)
        elos   = elos - elos.min() + 1.0
        probs  = elos / elos.sum()
        member = np.random.choice(len(self.members), p=probs)
        return self.members[member]["agent"]

    def __len__(self):
        return len(self.members)


# ── Multi-Agent Trainer ─────────────────────────────────────────────────────
class MultiAgentTrainer:
    """
    All 6 seats learn with shared weights via one AlphaPokerNet.
    Each seat maintains its own rollout buffer and opponent history.
    """
    def __init__(self, cfg: dict = CFG):
        self.cfg = cfg
        self.env = PokerEnv(
            num_players    = cfg["num_players"],
            starting_stack = cfg["starting_stack"],
            big_blind      = cfg["big_blind"],
            render_mode    = "none",
        )
        self.num_players = cfg["num_players"]

        # One shared agent (AlphaPokerNet)
        self.agent = PPOAgent(
            lr=cfg["lr"], clip_eps=cfg["clip_eps"],
            entropy_coef=cfg["entropy_coef"],
            value_coef=cfg["value_coef"],
            use_alpha=True,
            num_opponents=self.num_players - 1,
        )

        # Per-seat rollout buffers
        self.buffers = [
            RolloutBuffer(cfg["rollout_steps"],
                          num_opponents=self.num_players - 1)
            for _ in range(self.num_players)
        ]

        # Opponent history: histories[observer][opponent_slot] = deque of event vectors
        self.histories = [
            [deque(maxlen=MAX_SEQ_LEN) for _ in range(self.num_players)]
            for _ in range(self.num_players)
        ]

        # In-hand action tracker
        self.recorder = InHandRecorder(num_players=self.num_players)

        # League
        self.league   = League(max_size=cfg["league_size"])
        self.save_dir = Path(cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.hand_num     = 0
        self.total_steps  = 0
        self.reward_hist  = [deque(maxlen=1000) for _ in range(self.num_players)]
        self.win_hist     = deque(maxlen=1000)
        self.best_elo     = 1500.0
        self.last_metrics = {}
        self.ppo_updates  = 0
        self.t_start      = None

        # Scripted agents for phase 3
        self._scripted_seats = {}  # seat_idx -> ScriptedAgent (or None)

        # Exploration noise — decays over training
        self.opp_epsilon = cfg.get("opp_epsilon", 0.15)
        self.opp_epsilon_decay = cfg.get("opp_epsilon_decay", 0.9999)

        # Add initial random opponent to league
        rng_agent = PPOAgent(use_alpha=True,
                             num_opponents=self.num_players - 1)
        rng_agent.elo = 1200.0
        self.league.add(rng_agent, tag="random_init")

    def _get_opp_context(self, observer_idx: int) -> tuple:
        """
        Build opponent history tensors for one agent.
        Returns (opp_events, opp_masks) as numpy arrays.
        opp_events: (num_opponents, MAX_SEQ_LEN, EVENT_DIM)
        opp_masks:  (num_opponents, MAX_SEQ_LEN) — True where padded
        """
        num_opp = self.num_players - 1
        opp_events = np.zeros((num_opp, MAX_SEQ_LEN, EVENT_DIM), dtype=np.float32)
        opp_masks  = np.ones((num_opp, MAX_SEQ_LEN), dtype=bool)  # True = padded

        opp_slot = 0
        for pidx in range(self.num_players):
            if pidx == observer_idx:
                continue
            history = self.histories[observer_idx][pidx]
            n = len(history)
            for j in range(min(n, MAX_SEQ_LEN)):
                opp_events[opp_slot, j] = history[-(j + 1)]  # reversed chronological
                opp_masks[opp_slot, j] = False  # not padded
            opp_slot += 1

        return opp_events, opp_masks

    def _record_hand_events(self):
        """After hand ends, build events for all observer-player pairs."""
        for obs_idx in range(self.num_players):
            for player_idx in range(self.num_players):
                if obs_idx == player_idx:
                    continue
                event = self.recorder.build_event(obs_idx, player_idx, self.env)
                self.histories[obs_idx][player_idx].append(event)

    def _fill_showdown_targets(self, hand_start_ptrs: list):
        """Fill auxiliary showdown targets in all buffers."""
        info = self.env._showdown_info
        if not info.get("occurred", False):
            return

        revealed = info.get("cards", {})
        if not revealed:
            return

        from .poker_env import cards_to_onehot

        for seat_idx in range(self.num_players):
            start = hand_start_ptrs[seat_idx]
            end = self.buffers[seat_idx].ptr
            if start >= end:
                continue

            opp_slot = 0
            for pidx in range(self.num_players):
                if pidx == seat_idx:
                    continue
                if pidx in revealed:
                    card_vec = cards_to_onehot(revealed[pidx])
                    for t in range(start, end):
                        idx = t % self.buffers[seat_idx].capacity
                        self.buffers[seat_idx].showdown_cards[idx, opp_slot] = card_vec
                        self.buffers[seat_idx].has_showdown[idx] = True
                opp_slot += 1

    def _setup_phase(self, phase: int):
        """Configure scripted agents for the given phase."""
        self._scripted_seats = {}
        if phase == 3:
            styles = ["nit", "fish", "maniac", "tight-aggressive"]
            # Assign scripted agents to half the non-hero seats
            n_scripted = min(3, self.num_players - 1)
            seats = random.sample(range(self.num_players), n_scripted)
            for s in seats:
                self._scripted_seats[s] = ScriptedAgent(random.choice(styles))

    def _run_hand(self, phase: int) -> dict:
        obs = self.env.reset()
        self.recorder.reset()
        done = False
        metrics = {}
        max_steps = 200  # safety limit

        # Record buffer positions at hand start for showdown targets
        hand_start_ptrs = [buf.ptr for buf in self.buffers]

        step_count = 0
        while not done and step_count < max_steps:
            step_count += 1
            pidx = self.env.current_player
            o = obs[pidx]
            mask = self.env.legal_mask()

            # Check if this seat uses a scripted agent (phase 3)
            scripted = self._scripted_seats.get(pidx)

            if scripted is not None:
                action, raise_frac, _, _, _ = scripted.get_action(o, mask)
                obs_next, rewards, done, info = self.env.step(action, raise_frac)
                self.recorder.record(pidx, action, raise_frac,
                                     self.env.pot, self.env.street_idx)
            else:
                # All learning seats use the shared agent
                opp_events, opp_masks = self._get_opp_context(pidx)
                action, raise_frac, log_prob, value, entropy = \
                    self.agent.get_action(o, mask,
                                          opp_events=opp_events,
                                          opp_masks=opp_masks)

                # Epsilon-greedy exploration for non-hero seats
                # Adds diversity so opponents don't all play identically
                if pidx != 0 and random.random() < self.opp_epsilon:
                    legal = self.env.legal_actions()
                    action = random.choice(legal)
                    raise_frac = random.uniform(0.0, 1.0)

                obs_next, rewards, done, info = self.env.step(action, raise_frac)

                # Record action for all observers
                self.recorder.record(pidx, action, raise_frac,
                                     self.env.pot, self.env.street_idx)

                # Store in this player's buffer (clip reward to prevent vloss explosion)
                reward = np.clip(rewards.get(pidx, 0.0), -10.0, 10.0)
                self.buffers[pidx].add(
                    o, action, raise_frac, log_prob, reward, value,
                    float(done), mask, opp_events, opp_masks
                )
                self.total_steps += 1

                # PPO update when buffer is full
                if self.buffers[pidx].full:
                    metrics = self.agent.update(
                        self.buffers[pidx],
                        ppo_epochs=self.cfg["ppo_epochs"],
                        batch_size=self.cfg["batch_size"],
                        hand_num=self.hand_num,
                    )
                    self.last_metrics = metrics
                    self.ppo_updates += 1

            obs = obs_next

        # Hand ended — record events and fill targets
        self._record_hand_events()
        self._fill_showdown_targets(hand_start_ptrs)

        # Decay exploration noise
        self.opp_epsilon *= self.opp_epsilon_decay

        # Track per-seat rewards
        for i in range(self.num_players):
            r = self.env._rewards.get(i, 0.0)
            self.reward_hist[i].append(r)

        hero_reward = self.env._rewards.get(0, 0.0)
        self.win_hist.append(1.0 if hero_reward > 0 else 0.0)
        self.hand_num += 1

        return {"reward": hero_reward, "metrics": metrics}

    # ── ELO evaluation ────────────────────────────────────────────────────
    def _eval_elo(self, n_hands: int = 500) -> float:
        if not self.league.members:
            return self.agent.elo

        eval_env = PokerEnv(
            num_players    = self.cfg["num_players"],
            starting_stack = self.cfg["starting_stack"],
            big_blind      = self.cfg["big_blind"],
        )
        hero_elo = self.agent.elo
        wins = losses = 0

        for _ in range(n_hands):
            obs  = eval_env.reset()
            done = False
            while not done:
                pidx = eval_env.current_player
                o    = obs[pidx]
                mask = eval_env.legal_mask()
                if pidx == 0:
                    action, rf, _, _, _ = self.agent.get_action(
                        o, mask, deterministic=True)
                else:
                    opp = self.league.sample()
                    action, rf, _, _, _ = opp.get_action(o, mask, deterministic=True)
                obs, rewards, done, _ = eval_env.step(action, rf)

            if rewards.get(0, 0.0) > 0:
                wins += 1
            else:
                losses += 1

        win_rate = wins / max(wins + losses, 1)
        opp_elo  = np.mean([m["elo"] for m in self.league.members])

        expected = 1 / (1 + 10 ** ((opp_elo - hero_elo) / 400))
        hero_elo += 32 * (win_rate - expected)
        self.agent.elo = hero_elo
        return hero_elo

    # ── Main training loop ────────────────────────────────────────────────
    def train(self):
        dline = "\u2550" * 70
        print(f"\n{dline}")
        print(f"  AlphaPoker Multi-Agent Training")
        print(f"  Device: {self.agent.device}")
        total = self.cfg['phase1_hands'] + self.cfg['phase2_hands'] + self.cfg['phase3_hands']
        print(f"  Total hands: {total:,}")
        print(f"  Seats: {self.num_players} (all learning, shared weights)")
        print(f"{dline}\n")

        phase_schedule = [
            (1, self.cfg["phase1_hands"],  "Pure self-play (all seats learn)"),
            (2, self.cfg["phase2_hands"],  "League play"),
            (3, self.cfg["phase3_hands"],  "Archetype fine-tune"),
        ]

        for phase, n_hands, phase_name in phase_schedule:
            sline = "\u2500" * 70
            print(f"\n{sline}")
            print(f"  Phase {phase}: {phase_name}  ({n_hands:,} hands)")
            print(f"{sline}", flush=True)

            self._setup_phase(phase)
            self.t_start = time.time()
            phase_start_hand = self.hand_num

            for h in range(n_hands):
                result = self._run_hand(phase)

                log_freq = self.cfg.get("log_freq", 25)
                if self.hand_num % log_freq == 0:
                    elapsed = time.time() - self.t_start
                    hands_done = self.hand_num - phase_start_hand
                    hps = hands_done / elapsed if elapsed > 0 else 0
                    pct = hands_done / n_hands * 100

                    # Average reward across all seats
                    avg_rewards = [
                        np.mean(rh) if rh else 0.0
                        for rh in self.reward_hist
                    ]
                    mean_reward = np.mean(avg_rewards)

                    stats = {
                        "episode":     self.hand_num,
                        "step":        self.total_steps,
                        "mean_reward": mean_reward,
                        "win_pct":     np.mean(self.win_hist) if self.win_hist else 0,
                        "bb_per_100":  mean_reward * 100,
                        "elo":         self.agent.elo,
                        "hands_per_sec": hps,
                        "phase_pct":   pct,
                        "ppo_updates": self.ppo_updates,
                        **self.last_metrics,
                    }
                    print(render_training_stats(stats), flush=True)

                if self.hand_num % self.cfg["eval_freq"] == 0:
                    old_elo = self.agent.elo
                    new_elo = self._eval_elo(self.cfg["eval_hands"])
                    gain    = new_elo - old_elo
                    print(f"\n  ELO eval: {old_elo:.0f} \u2192 {new_elo:.0f}  ({gain:+.1f})", flush=True)

                    if phase >= 2 and gain >= self.cfg["min_elo_gain"]:
                        self.league.add(
                            self.agent,
                            tag=f"hand_{self.hand_num}_elo_{new_elo:.0f}"
                        )

                    if new_elo > self.best_elo:
                        self.best_elo = new_elo
                        self.agent.save(str(self.save_dir / "best.pt"))

                if self.hand_num % self.cfg["save_freq"] == 0:
                    path = self.save_dir / f"ckpt_hand_{self.hand_num:08d}.pt"
                    self.agent.save(str(path))

                if self.hand_num % self.cfg["render_every"] == 0:
                    self.env.render_mode = "unicode"
                    self._run_hand(phase)
                    self.env.render_mode = "none"

        dline = "\u2550" * 70
        print(f"\n{dline}")
        print(f"  Training complete!  Final ELO: {self.agent.elo:.0f}")
        print(f"  Best ELO achieved:  {self.best_elo:.0f}")
        print(f"  Checkpoints saved to: {self.save_dir}/")
        self.agent.save(str(self.save_dir / "final.pt"))


# ── Backward-compatible alias ────────────────────────────────────────────────
SelfPlayTrainer = MultiAgentTrainer


def main():
    """Entry point for `python -m pokerStats.rl.self_play_trainer`."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--phase",  type=int, default=None, help="Start at specific phase")
    parser.add_argument("--render", action="store_true", help="Show Unicode table every hand")
    args = parser.parse_args()

    trainer = MultiAgentTrainer(CFG)

    if args.resume:
        trainer.agent.load(args.resume)

    if args.render:
        trainer.env.render_mode = "unicode"
        CFG["render_every"] = 1

    trainer.train()


if __name__ == "__main__":
    main()
