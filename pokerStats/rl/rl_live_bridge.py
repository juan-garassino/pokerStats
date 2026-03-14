"""
RL -> Live bridge
──────────────────
Loads a trained PPO checkpoint and provides a Decision interface
for integration with the live autonomous agent.
"""

import numpy as np
from dataclasses import dataclass

from .poker_env import (
    Action, NUM_ACTIONS, OBS_DIM, CARD_IDX, cards_to_onehot, STREETS,
    hand_strength, draw_potential, raise_frac_to_amount,
)
from .ppo_agent import PPOAgent


# Stub Decision/OpponentProfile if CV modules not yet available
try:
    from pokerStats.decision_engine import Decision
except ImportError:
    @dataclass
    class Decision:
        action:     str
        amount:     float
        reasoning:  str
        confidence: float

try:
    from pokerStats.hand_history_db import OpponentProfile
except ImportError:
    @dataclass
    class OpponentProfile:
        name:             str = ""
        hands_played:     int = 0
        vpip:             float = 0.0
        pfr:              float = 0.0
        three_bet:        float = 0.0
        fold_to_3bet:     float = 0.0
        cbet_flop:        float = 0.0
        fold_to_cbet:     float = 0.0
        bluffs_seen:      int = 0
        bluffs_caught:    int = 0
        total_showdowns:  int = 0
        avg_bluff_size:   float = 0.0
        avg_value_size:   float = 0.0


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
        v = np.zeros(OBS_DIM, dtype=np.float32)

        # [0:52]  hole cards
        v[0:52]  = cards_to_onehot(hole_cards)

        # [52:104] community cards
        v[52:104] = cards_to_onehot(community_cards)

        # [104:110] player stacks (hero = idx 0, opponents fill 1-5)
        max_stack = 400.0
        v[104]    = hero_stack / max_stack
        profiles  = list(opponent_profiles.values())
        for i, p in enumerate(profiles[:5]):
            v[105+i] = p.hands_played / 1000.0

        # [110:116] current bets
        v[110]    = call_amount / max_stack

        # [116:122] active flags
        v[116]    = 1.0
        for i in range(min(num_opponents, 5)):
            v[117+i] = 1.0

        # [122:128] positions
        pos_map  = {"btn":0, "sb":1, "bb":2, "utg":3, "mp":4, "co":5, "hj":5}
        hero_pos = pos_map.get(str(position).lower(), 3)
        v[122]   = hero_pos / 6.0

        # [128:132] scalars
        v[128]   = pot / max_stack
        v[129]   = call_amount / max_stack
        v[130]   = call_amount / max_stack
        v[131]   = num_opponents / 5.0

        # [132:136] street one-hot
        street_idx = STREETS.index(street) if street in STREETS else 0
        v[132 + street_idx] = 1.0

        # [143] hand strength
        v[143] = hand_strength(hole_cards, community_cards)

        # [144] draw potential
        v[144] = draw_potential(hole_cards, community_cards)

        return v

    def _legal_mask(self, available_actions: list, call_amount: float) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        action_map = {
            "fold":  Action.FOLD,
            "check": Action.CHECK,
            "call":  Action.CALL,
            "raise": Action.RAISE,
            "allin": Action.ALL_IN,
        }
        for a in available_actions:
            idx = action_map.get(a)
            if idx is not None:
                mask[idx] = True

        if call_amount == 0 and not mask[Action.CHECK]:
            mask[Action.CHECK] = True
        if call_amount > 0:
            mask[Action.FOLD] = True

        return mask

    def _action_to_decision(
        self, action_idx: int, raise_frac: float,
        call_amount: float, pot: float, hero_stack: float
    ) -> Decision:
        if action_idx == Action.FOLD:
            act_name, amount = "fold", 0.0
        elif action_idx == Action.CHECK:
            act_name, amount = "check", 0.0
        elif action_idx == Action.CALL:
            act_name, amount = "call", call_amount
        elif action_idx == Action.RAISE:
            min_raise = call_amount + pot * 0.33
            max_raise = hero_stack
            amount = raise_frac_to_amount(raise_frac, min_raise, max_raise)
            act_name = "raise"
        elif action_idx == Action.ALL_IN:
            act_name, amount = "allin", hero_stack
        else:
            act_name, amount = "fold", 0.0

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
        action_idx, raise_frac, _, _, _ = self.agent.get_action(obs, mask, deterministic=False)
        return self._action_to_decision(action_idx, raise_frac, call_amount, pot, hero_stack)
