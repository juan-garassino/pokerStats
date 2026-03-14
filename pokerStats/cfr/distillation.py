"""
Blueprint teacher for CFR distillation into AlphaPokerNet
──────────────────────────────────────────────────────────
Wraps BlueprintStrategy + HandAbstraction to produce CFR target
probability vectors during PPO training.
"""

import numpy as np
from typing import Tuple

from .abstraction import HandAbstraction
from .blueprint import BlueprintStrategy
from .cfr_solver import NUM_CFR_ACTIONS

STREETS_BY_NCARDS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


class BlueprintTeacher:
    """Provides CFR blueprint targets during PPO training."""

    def __init__(self, blueprint_path: str, abstraction_path: str):
        self.blueprint = BlueprintStrategy()
        self.blueprint.load(blueprint_path)
        self.abstraction = HandAbstraction()
        self.abstraction.load(abstraction_path)

    def get_target(
        self,
        hole_cards: list,
        community: list,
        action_history: tuple,
    ) -> Tuple[np.ndarray, bool]:
        """
        Look up the CFR blueprint strategy for the given game state.

        Args:
            hole_cards: Hero's hole cards, e.g. ["Ah", "Ks"]
            community: Community cards, e.g. ["Qh", "Jd", "2c"]
            action_history: Tuple of CFR action indices taken so far

        Returns:
            (cfr_probs, found) where cfr_probs is a (NUM_CFR_ACTIONS,) array
            and found indicates whether the blueprint had an entry.
        """
        bucket = self.abstraction.get_bucket(hole_cards, community)
        street = STREETS_BY_NCARDS.get(len(community), "preflop")
        probs = self.blueprint.lookup(bucket, street, action_history)
        if probs is not None:
            return probs.astype(np.float32), True
        return np.zeros(NUM_CFR_ACTIONS, dtype=np.float32), False

    @property
    def num_cfr_actions(self) -> int:
        return NUM_CFR_ACTIONS
