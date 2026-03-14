"""
Self-play + league training
────────────────────────────
Training curriculum:
  Phase 1 — Pure self-play (agent vs copies of itself)
  Phase 2 — League play (agent vs pool of frozen past checkpoints)
  Phase 3 — Fine-tune vs human-mimic opponents (scripted archetypes)
"""

import copy
import random
import numpy as np
from pathlib import Path
from collections import deque

from .poker_env import PokerEnv, NUM_ACTIONS
from .ppo_agent import PPOAgent, RolloutBuffer
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

    # Self-play
    "self_play_update_freq":  500,
    "league_size":            8,
    "league_add_freq":        2000,
    "eval_freq":              1000,
    "eval_hands":             500,
    "min_elo_gain":           10.0,

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
                   deterministic: bool = False) -> tuple:
        legal = [i for i in range(NUM_ACTIONS) if legal_mask[i]]
        if not legal:
            return 0, 0.0, 0.0, 0.0

        # Decode obs scalars
        call_norm   = float(obs[129])
        street_oh   = obs[132:136]
        street_idx  = int(np.argmax(street_oh)) if any(street_oh) else 0
        call_high   = call_norm > 0.05

        if self.style == "nit":
            if call_high and 5 not in legal:
                action = 0
            elif not call_high:
                action = 1
            else:
                action = 2

        elif self.style == "fish":
            action = 2 if 2 in legal else (1 if 1 in legal else 0)

        elif self.style == "maniac":
            if 6 in legal and random.random() < 0.3:
                action = 6
            elif 5 in legal and random.random() < 0.4:
                action = 5
            elif 4 in legal:
                action = 4
            else:
                action = random.choice(legal)

        elif self.style == "tight-aggressive":
            if street_idx == 0:
                if call_high:
                    action = 4 if 4 in legal and random.random() < 0.5 else 2
                else:
                    action = 3 if 3 in legal else 1
            else:
                action = random.choice(legal)
        else:
            action = random.choice(legal)

        return int(action), 0.0, 0.0, 0.0


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


# ── Trainer ───────────────────────────────────────────────────────────────────
class SelfPlayTrainer:
    def __init__(self, cfg: dict = CFG):
        self.cfg     = cfg
        self.env     = PokerEnv(
            num_players    = cfg["num_players"],
            starting_stack = cfg["starting_stack"],
            big_blind      = cfg["big_blind"],
            render_mode    = "none",
        )
        self.agent   = PPOAgent(lr=cfg["lr"], clip_eps=cfg["clip_eps"],
                                entropy_coef=cfg["entropy_coef"],
                                value_coef=cfg["value_coef"])
        self.buffer  = RolloutBuffer(cfg["rollout_steps"])
        self.league  = League(max_size=cfg["league_size"])
        self.save_dir= Path(cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.hand_num   = 0
        self.total_steps= 0
        self.reward_hist= deque(maxlen=1000)
        self.win_hist   = deque(maxlen=1000)
        self.best_elo   = 1500.0

        # Add initial random opponent to league
        rng_agent = PPOAgent()
        rng_agent.elo = 1200.0
        self.league.add(rng_agent, tag="random_init")

    # ── Opponent pool ──────────────────────────────────────────────────────────
    def _get_opponents(self, phase: int) -> list:
        n_opp = self.cfg["num_players"] - 1

        if phase == 1:
            return [self.agent] * n_opp

        elif phase == 2:
            opponents = []
            for _ in range(n_opp):
                league_opp = self.league.sample_weighted()
                if league_opp and random.random() < 0.5:
                    opponents.append(league_opp)
                else:
                    opponents.append(self.agent)
            return opponents

        else:  # phase 3
            styles = ["nit", "fish", "maniac", "tight-aggressive"]
            opponents = []
            for i in range(n_opp):
                r = random.random()
                if r < 0.3 and self.league.members:
                    opponents.append(self.league.sample_weighted())
                elif r < 0.6:
                    opponents.append(ScriptedAgent(random.choice(styles)))
                else:
                    opponents.append(self.agent)
            return opponents

    # ── Run one hand ──────────────────────────────────────────────────────────
    def _run_hand(self, phase: int) -> dict:
        obs      = self.env.reset()
        opponents= self._get_opponents(phase)
        done     = False
        metrics  = {}

        while not done:
            pidx  = self.env.current_player
            o     = obs[pidx]
            mask  = self.env.legal_mask()

            if pidx == 0:  # hero (learning agent)
                action, log_prob, value, entropy = self.agent.get_action(o, mask)
                obs_next, rewards, done, info = self.env.step(action)

                reward = rewards.get(0, 0.0)
                self.buffer.add(o, action, log_prob, reward, value,
                                float(done), mask)
                self.total_steps += 1

                if self.buffer.full:
                    metrics = self.agent.update(
                        self.buffer,
                        ppo_epochs=self.cfg["ppo_epochs"],
                        batch_size=self.cfg["batch_size"],
                    )
            else:
                opp_idx = pidx - 1
                opp     = opponents[min(opp_idx, len(opponents)-1)]
                action, _, _, _ = opp.get_action(o, mask)
                obs_next, rewards, done, info = self.env.step(action)

            obs = obs_next

        hero_reward = self.env._rewards.get(0, 0.0)
        self.reward_hist.append(hero_reward)
        self.win_hist.append(1.0 if hero_reward > 0 else 0.0)
        self.hand_num += 1

        return {
            "reward":  hero_reward,
            "metrics": metrics,
        }

    # ── ELO evaluation ────────────────────────────────────────────────────────
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
                    action, _, _, _ = self.agent.get_action(o, mask, deterministic=True)
                else:
                    opp    = self.league.sample()
                    action, _, _, _ = opp.get_action(o, mask, deterministic=True)
                obs, rewards, done, _ = eval_env.step(action)

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

    # ── Main training loop ────────────────────────────────────────────────────
    def train(self):
        dline = "\u2550" * 70
        print(f"\n{dline}")
        print(f"  PokerStars RL Training \u2014 Self-Play + League")
        print(f"  Device: {self.agent.device}")
        total = self.cfg['phase1_hands'] + self.cfg['phase2_hands'] + self.cfg['phase3_hands']
        print(f"  Total hands: {total:,}")
        print(f"{dline}\n")

        phase_schedule = [
            (1, self.cfg["phase1_hands"],  "Pure self-play"),
            (2, self.cfg["phase2_hands"],  "League play"),
            (3, self.cfg["phase3_hands"],  "Human archetype fine-tune"),
        ]

        for phase, n_hands, phase_name in phase_schedule:
            sline = "\u2500" * 70
            print(f"\n{sline}")
            print(f"  Phase {phase}: {phase_name}  ({n_hands:,} hands)")
            print(sline)

            for h in range(n_hands):
                result = self._run_hand(phase)

                if self.hand_num % 100 == 0:
                    stats = {
                        "episode":     self.hand_num,
                        "step":        self.total_steps,
                        "mean_reward": np.mean(self.reward_hist) if self.reward_hist else 0,
                        "win_pct":     np.mean(self.win_hist)    if self.win_hist   else 0,
                        "bb_per_100":  np.mean(self.reward_hist)*100 if self.reward_hist else 0,
                        "elo":         self.agent.elo,
                        **result.get("metrics", {}),
                    }
                    print(render_training_stats(stats))

                if self.hand_num % self.cfg["eval_freq"] == 0:
                    old_elo = self.agent.elo
                    new_elo = self._eval_elo(self.cfg["eval_hands"])
                    gain    = new_elo - old_elo
                    print(f"\n  ELO eval: {old_elo:.0f} \u2192 {new_elo:.0f}  ({gain:+.1f})")

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


def main():
    """Entry point for `python -m pokerStats.rl.self_play_trainer`."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--phase",  type=int, default=None, help="Start at specific phase")
    parser.add_argument("--render", action="store_true", help="Show Unicode table every hand")
    args = parser.parse_args()

    trainer = SelfPlayTrainer(CFG)

    if args.resume:
        trainer.agent.load(args.resume)

    if args.render:
        trainer.env.render_mode = "unicode"
        CFG["render_every"] = 1

    trainer.train()


if __name__ == "__main__":
    main()
