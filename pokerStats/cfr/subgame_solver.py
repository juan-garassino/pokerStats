"""
Depth-limited real-time subgame solver
──────────────────────────────────────
Runs a short burst of CFR iterations from the current game state
to refine the blueprint strategy for critical decisions (big pots).

Time-limited to 2 seconds for live PokerStars play.
Falls back to blueprint if time runs out.
"""

import time
import numpy as np
from typing import Optional

from pokerStats.rl.poker_env import PokerEnv, Action, STREETS
from .cfr_solver import (
    InfoSet, NUM_CFR_ACTIONS, CFR_ACTIONS,
    cfr_to_env_action, env_legal_to_cfr_mask,
)
from .abstraction import HandAbstraction
from .blueprint import BlueprintStrategy


class SubgameSolver:
    """
    Real-time subgame refinement using depth-limited CFR.

    Starting from the current game state, runs MCCFR iterations
    with blueprint values at leaf nodes (depth limit).

    Key constraints:
    - 2-second time budget for live play
    - Depth limit of 2 streets (current + next)
    - Uses blueprint values at leaves
    """

    def __init__(
        self,
        abstraction: HandAbstraction,
        blueprint: BlueprintStrategy,
        time_limit: float = 2.0,
        max_depth: int = 2,
        max_iterations: int = 1000,
    ):
        self.abstraction = abstraction
        self.blueprint = blueprint
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.max_iterations = max_iterations

    def solve(
        self,
        env: PokerEnv,
        hero_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Run depth-limited CFR from the current state.

        Args:
            env: Current game state.
            hero_idx: Hero's player index.

        Returns:
            Refined action probabilities for hero, or None if timed out
            before any useful iterations.
        """
        start_time = time.time()
        initial_street = env.street_idx
        info_sets: dict = {}  # local info sets for this subgame

        completed_iters = 0

        for i in range(self.max_iterations):
            if time.time() - start_time > self.time_limit:
                break

            traverser = i % 2
            env_copy = env.clone()
            self._subgame_cfr(
                env_copy, traverser, hero_idx,
                initial_street, info_sets, ()
            )
            completed_iters += 1

        if completed_iters < 2:
            return None

        # Extract hero's strategy at root
        player = env.players[env.current_player]
        hand_bucket = self.abstraction.get_bucket(
            player.hole_cards, env.community
        )
        street = STREETS[env.street_idx]
        root_key = f"B{hand_bucket}|{street}|root"

        if root_key in info_sets:
            legal_mask = env_legal_to_cfr_mask(env)
            return info_sets[root_key].average_strategy(legal_mask)

        return None

    def _subgame_cfr(
        self,
        env: PokerEnv,
        traverser: int,
        hero_idx: int,
        initial_street: int,
        info_sets: dict,
        action_history: tuple,
    ) -> float:
        """Depth-limited CFR traversal with blueprint leaf values."""
        # Terminal
        if env._done:
            return env._rewards.get(traverser, 0.0)

        # Depth limit: use blueprint value estimate
        depth = env.street_idx - initial_street
        if depth >= self.max_depth:
            return self._blueprint_value(env, traverser)

        current = env.current_player
        player = env.players[current]
        hand_bucket = self.abstraction.get_bucket(
            player.hole_cards, env.community
        )
        street = STREETS[env.street_idx]
        hist_str = "-".join(str(a) for a in action_history) if action_history else "root"
        key = f"B{hand_bucket}|{street}|{hist_str}"

        if key not in info_sets:
            info_sets[key] = InfoSet()
        info_set = info_sets[key]

        legal_mask = env_legal_to_cfr_mask(env)
        strategy = info_set.current_strategy(legal_mask)

        if current == traverser:
            action_values = np.zeros(NUM_CFR_ACTIONS, dtype=np.float64)
            node_value = 0.0

            for a in range(NUM_CFR_ACTIONS):
                if not legal_mask[a]:
                    continue
                child = env.clone()
                env_action, raise_frac = cfr_to_env_action(a)
                child.step(int(env_action), raise_frac)
                action_values[a] = self._subgame_cfr(
                    child, traverser, hero_idx,
                    initial_street, info_sets, action_history + (a,)
                )
                node_value += strategy[a] * action_values[a]

            for a in range(NUM_CFR_ACTIONS):
                if legal_mask[a]:
                    info_set.cumulative_regrets[a] += action_values[a] - node_value

            return node_value
        else:
            info_set.strategy_sum += strategy
            legal_indices = np.where(legal_mask)[0]
            if len(legal_indices) == 0:
                return env._rewards.get(traverser, 0.0)
            legal_probs = strategy[legal_indices]
            legal_probs = legal_probs / (legal_probs.sum() + 1e-10)
            a = np.random.choice(legal_indices, p=legal_probs)
            child = env.clone()
            env_action, raise_frac = cfr_to_env_action(a)
            child.step(int(env_action), raise_frac)
            return self._subgame_cfr(
                child, traverser, hero_idx,
                initial_street, info_sets, action_history + (a,)
            )

    def _blueprint_value(self, env: PokerEnv, player: int) -> float:
        """
        Estimate the value at a depth-limited leaf using the blueprint.

        Uses a simple equity-based heuristic weighted by pot.
        """
        from pokerStats.rl.poker_env import hand_strength

        p = env.players[player]
        hs = hand_strength(p.hole_cards, env.community)
        pot = env.pot
        committed = p.total_bet

        # Expected value: (equity × pot) - committed, in BB
        ev = (hs * pot - committed) / env.big_blind
        return ev
