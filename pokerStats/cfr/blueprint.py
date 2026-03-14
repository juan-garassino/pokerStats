"""
Blueprint strategy storage
───────────────────────────
Extracts and compresses the average strategies from a trained CFR solver
into a fast-lookup format for live play.

Target: <500MB on disk using float16 compression.
"""

import numpy as np
from typing import Optional

from .cfr_solver import CFRSolver, InfoSet, NUM_CFR_ACTIONS


class BlueprintStrategy:
    """
    Compressed strategy table extracted from a trained CFR solver.

    Stores the converged average strategies (Nash approximation)
    as float16 arrays indexed by info set key.
    """

    def __init__(self):
        self.strategies: dict = {}  # key → np.ndarray (float16, NUM_CFR_ACTIONS)

    @classmethod
    def from_solver(cls, solver: CFRSolver, min_visits: int = 10) -> 'BlueprintStrategy':
        """
        Extract average strategies from a trained solver.

        Args:
            solver: Trained CFRSolver with populated info_sets.
            min_visits: Minimum strategy_sum total to include (filters noise).

        Returns:
            BlueprintStrategy with compressed float16 strategies.
        """
        bp = cls()
        for key, info_set in solver.info_sets.items():
            total = info_set.strategy_sum.sum()
            if total < min_visits:
                continue
            avg = info_set.average_strategy()
            bp.strategies[key] = avg.astype(np.float16)
        return bp

    def lookup(
        self,
        hand_bucket: int,
        street: str,
        action_history: tuple,
    ) -> Optional[np.ndarray]:
        """
        Look up the blueprint strategy for an information set.

        Args:
            hand_bucket: Abstraction bucket ID.
            street: Current street name.
            action_history: Tuple of CFR action indices.

        Returns:
            Action probability array (float64) or None if not found.
        """
        hist_str = "-".join(str(a) for a in action_history) if action_history else "root"
        key = f"B{hand_bucket}|{street}|{hist_str}"
        if key in self.strategies:
            return self.strategies[key].astype(np.float64)
        return None

    def __len__(self) -> int:
        return len(self.strategies)

    def __contains__(self, key: str) -> bool:
        return key in self.strategies

    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        n_entries = len(self.strategies)
        # Each entry: key string (~50 bytes) + float16 array (NUM_CFR_ACTIONS * 2 bytes)
        bytes_per_entry = 50 + NUM_CFR_ACTIONS * 2
        return n_entries * bytes_per_entry / (1024 * 1024)

    def save(self, path: str):
        """Save blueprint to compressed numpy archive."""
        keys = np.array(list(self.strategies.keys()), dtype=object)
        values = np.array(list(self.strategies.values()), dtype=np.float16)
        np.savez_compressed(path, keys=keys, values=values)

    def load(self, path: str):
        """Load blueprint from compressed numpy archive."""
        data = np.load(path, allow_pickle=True)
        keys = data["keys"]
        values = data["values"]
        self.strategies = {}
        for k, v in zip(keys, values):
            self.strategies[str(k)] = v
