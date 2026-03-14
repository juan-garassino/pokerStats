"""
RL → Live bridge
──────────────────
Loads a trained PPO checkpoint and replaces the stub DecisionEngine
in autonomous_agent.py with the real RL policy.

Usage:
  python rl_live_bridge.py --model checkpoints/best.pt --dry-run
"""

import argparse
import numpy as np

from poker_env import PokerEnv, Action, NUM_ACTIONS, OBS_DIM, CARD_IDX, cards_to_onehot, STREETS
from ppo_agent import PPOAgent
from decision_engine import Decision
from hand_history_db import OpponentProfile


class RLDecisionEngine:
    """
    Drop-in replacement for decision_engine.DecisionEngine.
    Uses the trained PPO policy instead of hand-crafted rules.
    """

    def __init__(self, checkpoint_path: str):
        self.agent = PPOAgent()
        self.agent.load(checkpoint_path)
        self.agent.net.eval()
        print(f"RL engine loaded (ELO={self.agent.elo:.0f})")

    def _build_obs(
        self,
        hole_cards:       list,
        community_cards:  list,
        street:           str,
        pot:              float,
        hero_stack:       float,
        call_amount:      float,
        position:         str,
        num_opponents:    int,
        opponent_profiles: dict,
    ) -> np.ndarray:
        """
        Build the 143-dim observation vector from live game state.
        Must exactly match the format used during training in poker_env.py.
        """
        v = np.zeros(OBS_DIM, dtype=np.float32)

        # [0:52]  hole cards
        v[0:52]  = cards_to_onehot(hole_cards)

        # [52:104] community cards
        v[52:104] = cards_to_onehot(community_cards)

        # [104:110] player stacks (hero = idx 0, opponents fill 1–5)
        max_stack = 400.0
        v[104]    = hero_stack / max_stack
        profiles  = list(opponent_profiles.values())
        for i, p in enumerate(profiles[:5]):
            v[105+i] = p.hands_played / 1000.0  # proxy for stack (not known live)

        # [110:116] current bets — hero bet = call_amount, opponents unknown
        v[110]    = call_amount / max_stack

        # [116:122] active flags — approximate
        v[116]    = 1.0  # hero
        for i in range(min(num_opponents, 5)):
            v[117+i] = 1.0

        # [122:128] positions
        pos_map  = {"btn":0, "sb":1, "bb":2, "utg":3, "mp":4, "co":5, "hj":5}
        hero_pos = pos_map.get(str(position).lower(), 3)
        v[122]   = hero_pos / 6.0

        # [128:132] scalars
        v[128]   = pot / max_stack
        v[129]   = call_amount / max_stack
        v[130]   = call_amount / max_stack   # min raise approx
        v[131]   = num_opponents / 5.0

        # [132:136] street one-hot
        street_idx = STREETS.index(street) if street in STREETS else 0
        v[132 + street_idx] = 1.0

        return v

    def _legal_mask(self, available_actions: list, call_amount: float) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        action_map = {
            "fold":  0,
            "check": 1,
            "call":  2,
            "raise": [3, 4, 5],
            "allin": 6,
        }
        for a in available_actions:
            idx = action_map.get(a)
            if idx is None:
                continue
            if isinstance(idx, list):
                for i in idx:
                    mask[i] = True
            else:
                mask[idx] = True

        # If no call needed, check replaces call
        if call_amount == 0 and not mask[1]:
            mask[1] = True

        # Fold always valid if there's a bet to call
        if call_amount > 0:
            mask[0] = True

        return mask

    def _action_to_decision(
        self, action_idx: int, call_amount: float, pot: float, hero_stack: float
    ) -> Decision:
        action_map = {
            Action.FOLD:     ("fold",  0.0),
            Action.CHECK:    ("check", 0.0),
            Action.CALL:     ("call",  call_amount),
            Action.RAISE_25: ("raise", call_amount + pot * 0.25),
            Action.RAISE_50: ("raise", call_amount + pot * 0.50),
            Action.RAISE_100:("raise", call_amount + pot * 1.00),
            Action.ALL_IN:   ("allin", hero_stack),
        }
        act_name, amount = action_map.get(action_idx, ("fold", 0.0))
        amount = min(amount, hero_stack)
        return Decision(action=act_name, amount=round(amount, 2),
                        reasoning=f"RL policy (ELO={self.agent.elo:.0f})",
                        confidence=0.9)

    def decide(
        self,
        hole_cards:        list,
        community_cards:   list,
        street:            str,
        pot:               float,
        hero_stack:        float,
        call_amount:       float,
        position:          str,
        num_opponents:     int,
        opponent_profiles: dict,
        available_actions: list,
    ) -> Decision:

        obs  = self._build_obs(hole_cards, community_cards, street,
                               pot, hero_stack, call_amount,
                               position, num_opponents, opponent_profiles)
        mask = self._legal_mask(available_actions, call_amount)

        action_idx, _, _, _ = self.agent.get_action(obs, mask, deterministic=False)
        return self._action_to_decision(action_idx, call_amount, pot, hero_stack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="checkpoints/best.pt")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--fps",      type=int, default=10)
    args = parser.parse_args()

    # Patch the autonomous agent to use RL engine
    import autonomous_agent as aa
    from autonomous_agent import AutonomousAgent

    engine = RLDecisionEngine(args.model)

    agent = AutonomousAgent(dry_run=args.dry_run, fps=args.fps)
    agent.engine = engine    # hot-swap the decision engine

    print(f"\nLive RL agent active  ELO={engine.agent.elo:.0f}  dry_run={args.dry_run}")
    agent.run()
