"""
External Sampling MCCFR solver
──────────────────────────────
Monte Carlo Counterfactual Regret Minimization with external sampling.

The most practical CFR variant for NLHE:
- Only traverses one player's decision nodes per iteration
- Opponent actions are sampled from current strategy (external sampling)
- Converges to Nash equilibrium in two-player zero-sum games
"""

import sys
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Deep raise-reraise sequences can exceed default recursion limit
sys.setrecursionlimit(10_000)

from pokerStats.rl.poker_env import (
    PokerEnv, Action, NUM_ACTIONS, STREETS,
)
from .abstraction import HandAbstraction


# ── CFR action abstraction ───────────────────────────────────────────────────
# Map CFR discrete actions to poker_env actions + raise fractions

CFR_ACTIONS = [
    (Action.FOLD, 0.0),       # 0: fold
    (Action.CHECK, 0.0),      # 1: check
    (Action.CALL, 0.0),       # 2: call
    (Action.RAISE, 0.15),     # 3: raise ~33% pot (frac 0.15 → ~12% of range via exp curve)
    (Action.RAISE, 0.45),     # 4: raise ~75% pot
    (Action.RAISE, 0.70),     # 5: raise ~150% pot
    (Action.ALL_IN, 0.0),     # 6: all-in
]
NUM_CFR_ACTIONS = len(CFR_ACTIONS)


def cfr_to_env_action(cfr_action: int) -> tuple:
    """Convert CFR action index to (Action, raise_frac) pair."""
    return CFR_ACTIONS[cfr_action]


def env_legal_to_cfr_mask(env: PokerEnv) -> np.ndarray:
    """Build a CFR-action-space legal mask from the env's legal actions."""
    env_legal = set(env.legal_actions())
    mask = np.zeros(NUM_CFR_ACTIONS, dtype=bool)
    for i, (action, _) in enumerate(CFR_ACTIONS):
        if int(action) in env_legal:
            mask[i] = True
    return mask


# ── InfoSet ──────────────────────────────────────────────────────────────────

@dataclass
class InfoSet:
    """
    Stores cumulative regrets and strategy sums for one information set.

    Regret matching produces the current strategy.
    The average strategy (strategy_sum normalized) converges to Nash.
    """
    num_actions: int = NUM_CFR_ACTIONS
    cumulative_regrets: np.ndarray = field(default=None)
    strategy_sum: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.cumulative_regrets is None:
            self.cumulative_regrets = np.zeros(self.num_actions, dtype=np.float64)
        if self.strategy_sum is None:
            self.strategy_sum = np.zeros(self.num_actions, dtype=np.float64)

    def current_strategy(self, legal_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Regret matching: strategy proportional to positive regrets.

        If all regrets are non-positive, returns uniform over legal actions.
        """
        strategy = np.maximum(self.cumulative_regrets, 0.0)

        if legal_mask is not None:
            strategy *= legal_mask

        total = strategy.sum()
        if total > 0:
            return strategy / total

        # Uniform over legal actions
        if legal_mask is not None:
            n_legal = legal_mask.sum()
            if n_legal > 0:
                return legal_mask.astype(np.float64) / n_legal
        return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

    def average_strategy(self, legal_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The converged Nash equilibrium strategy.

        This is the output used in the blueprint — the average of all
        strategies played across all iterations.
        """
        avg = self.strategy_sum.copy()
        if legal_mask is not None:
            avg *= legal_mask
        total = avg.sum()
        if total > 0:
            return avg / total
        if legal_mask is not None:
            n_legal = legal_mask.sum()
            if n_legal > 0:
                return legal_mask.astype(np.float64) / n_legal
        return np.ones(self.num_actions, dtype=np.float64) / self.num_actions


# ── CFR Solver ───────────────────────────────────────────────────────────────

class CFRSolver:
    """
    External Sampling MCCFR for heads-up NLHE.

    Uses hand abstraction to keep the info set space tractable.
    Alternates which player is the "traverser" each iteration.
    """

    MAX_DEPTH = 50  # max actions per hand before forced evaluation

    def __init__(self, abstraction: Optional[HandAbstraction] = None):
        self.abstraction = abstraction or HandAbstraction()
        self.info_sets: dict = {}  # key → InfoSet
        self.iterations = 0

    def _info_set_key(
        self,
        hand_bucket: int,
        street: str,
        action_history: tuple,
    ) -> str:
        """
        Build a unique key for an information set.

        Format: "B{bucket}|{street}|{action_history}"
        Example: "B42|flop|c-r75-c"
        """
        hist_str = "-".join(str(a) for a in action_history) if action_history else "root"
        return f"B{hand_bucket}|{street}|{hist_str}"

    def _get_info_set(self, key: str) -> InfoSet:
        """Get or create an InfoSet for the given key."""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSet()
        return self.info_sets[key]

    def external_sampling_cfr(
        self,
        env: PokerEnv,
        traversing_player: int,
        action_history: tuple = (),
    ) -> float:
        """
        Recursive external sampling MCCFR traversal.

        Args:
            env: Current game state (will be cloned for branching).
            traversing_player: The player whose regrets we're updating.
            action_history: Tuple of CFR action indices taken so far.

        Returns:
            Counterfactual value for the traversing player.
        """
        # Terminal check
        if env._done:
            return env._rewards.get(traversing_player, 0.0)

        # Depth limit: treat excessively long action sequences as terminal
        if len(action_history) >= self.MAX_DEPTH:
            return env._rewards.get(traversing_player, 0.0)

        current = env.current_player
        player = env.players[current]

        # Get hand bucket for current player
        hand_bucket = self.abstraction.get_bucket(
            player.hole_cards, env.community
        )
        street = STREETS[env.street_idx]
        key = self._info_set_key(hand_bucket, street, action_history)
        info_set = self._get_info_set(key)

        # Legal action mask in CFR action space
        legal_mask = env_legal_to_cfr_mask(env)

        if current == traversing_player:
            # ── Traverser node: compute regrets for ALL actions ──────────
            strategy = info_set.current_strategy(legal_mask)
            action_values = np.zeros(NUM_CFR_ACTIONS, dtype=np.float64)
            node_value = 0.0

            for a in range(NUM_CFR_ACTIONS):
                if not legal_mask[a]:
                    continue

                child_env = env.clone()
                env_action, raise_frac = cfr_to_env_action(a)
                child_env.step(int(env_action), raise_frac)

                action_values[a] = self.external_sampling_cfr(
                    child_env, traversing_player, action_history + (a,)
                )
                node_value += strategy[a] * action_values[a]

            # Update regrets
            for a in range(NUM_CFR_ACTIONS):
                if legal_mask[a]:
                    regret = action_values[a] - node_value
                    info_set.cumulative_regrets[a] += regret

            return node_value

        else:
            # ── Opponent node: sample ONE action from current strategy ──
            strategy = info_set.current_strategy(legal_mask)

            # Accumulate strategy sum (for average strategy computation)
            info_set.strategy_sum += strategy

            # Sample action
            legal_indices = np.where(legal_mask)[0]
            if len(legal_indices) == 0:
                return env._rewards.get(traversing_player, 0.0)

            legal_probs = strategy[legal_indices]
            legal_probs = legal_probs / (legal_probs.sum() + 1e-10)
            sampled_idx = np.random.choice(legal_indices, p=legal_probs)

            child_env = env.clone()
            env_action, raise_frac = cfr_to_env_action(sampled_idx)
            child_env.step(int(env_action), raise_frac)

            return self.external_sampling_cfr(
                child_env, traversing_player, action_history + (sampled_idx,)
            )

    def train(self, n_iterations: int, log_every: int = 1000):
        """
        Run external sampling MCCFR for n_iterations.

        Each iteration creates a fresh 2-player environment and alternates
        which player is the traverser.
        """
        env = PokerEnv(num_players=2, render_mode="none")

        for i in range(n_iterations):
            traverser = i % 2
            env.reset()
            self.external_sampling_cfr(env, traverser)
            self.iterations += 1

            if log_every > 0 and (i + 1) % log_every == 0:
                print(f"  CFR iteration {self.iterations}, "
                      f"info sets: {len(self.info_sets)}")

    def exploitability_estimate(self, n_samples: int = 1000) -> float:
        """
        Estimate exploitability by computing best-response value.

        Lower exploitability = closer to Nash equilibrium.
        Returns approximate exploitability in BB/hand.
        """
        env = PokerEnv(num_players=2, render_mode="none")
        total_exploit = 0.0

        for _ in range(n_samples):
            env.reset()
            # Compute best response value for player 0 against player 1's avg strategy
            br_val = self._best_response_value(env, 0, ())
            total_exploit += br_val

        return total_exploit / n_samples

    def _best_response_value(
        self,
        env: PokerEnv,
        br_player: int,
        action_history: tuple,
    ) -> float:
        """Compute best-response value for br_player against opponent's average strategy."""
        if env._done:
            return env._rewards.get(br_player, 0.0)

        if len(action_history) >= self.MAX_DEPTH:
            return env._rewards.get(br_player, 0.0)

        current = env.current_player
        player = env.players[current]
        hand_bucket = self.abstraction.get_bucket(player.hole_cards, env.community)
        street = STREETS[env.street_idx]
        key = self._info_set_key(hand_bucket, street, action_history)
        legal_mask = env_legal_to_cfr_mask(env)

        if current == br_player:
            # Best response: take the action with highest value
            best_val = float("-inf")
            for a in range(NUM_CFR_ACTIONS):
                if not legal_mask[a]:
                    continue
                child_env = env.clone()
                env_action, raise_frac = cfr_to_env_action(a)
                child_env.step(int(env_action), raise_frac)
                val = self._best_response_value(child_env, br_player, action_history + (a,))
                best_val = max(best_val, val)
            return best_val if best_val > float("-inf") else 0.0

        else:
            # Opponent plays average strategy
            info_set = self._get_info_set(key)
            strategy = info_set.average_strategy(legal_mask)
            node_value = 0.0
            for a in range(NUM_CFR_ACTIONS):
                if not legal_mask[a] or strategy[a] < 1e-10:
                    continue
                child_env = env.clone()
                env_action, raise_frac = cfr_to_env_action(a)
                child_env.step(int(env_action), raise_frac)
                val = self._best_response_value(child_env, br_player, action_history + (a,))
                node_value += strategy[a] * val
            return node_value

    def save(self, path: str):
        """Save all info sets to a compressed numpy archive."""
        keys = []
        regrets = []
        sums = []
        for key, info_set in self.info_sets.items():
            keys.append(key)
            regrets.append(info_set.cumulative_regrets)
            sums.append(info_set.strategy_sum)

        np.savez_compressed(
            path,
            keys=np.array(keys, dtype=object),
            regrets=np.array(regrets),
            sums=np.array(sums),
            iterations=np.array(self.iterations),
        )

    def load(self, path: str):
        """Load info sets from a compressed numpy archive."""
        data = np.load(path, allow_pickle=True)
        keys = data["keys"]
        regrets = data["regrets"]
        sums = data["sums"]
        self.iterations = int(data["iterations"])

        self.info_sets = {}
        for i, key in enumerate(keys):
            info_set = InfoSet()
            info_set.cumulative_regrets = regrets[i].copy()
            info_set.strategy_sum = sums[i].copy()
            self.info_sets[str(key)] = info_set


# ── Kuhn Poker (for testing) ────────────────────────────────────────────────

class KuhnPokerEnv:
    """
    Minimal Kuhn poker implementation for CFR correctness testing.

    3 cards (J, Q, K), 2 players, 1 betting round.
    Known Nash equilibrium — if CFR converges to it, the solver is correct.

    Nash equilibrium for player 0:
      J: bet ~1/3 (bluff), check-call never, check-fold always after opponent bet
      Q: check always, call ~1/3 after opponent bet
      K: bet ~3× check frequency, always call
    """

    def __init__(self):
        self.reset()

    def reset(self):
        cards = [0, 1, 2]  # J=0, Q=1, K=2
        random.shuffle(cards)
        self.cards = cards[:2]  # one per player
        self.history = []
        self.current_player = 0
        self._done = False
        self._rewards = {0: 0.0, 1: 0.0}
        return self

    def clone(self):
        env = KuhnPokerEnv.__new__(KuhnPokerEnv)
        env.cards = list(self.cards)
        env.history = list(self.history)
        env.current_player = self.current_player
        env._done = self._done
        env._rewards = dict(self._rewards)
        return env

    def legal_actions(self) -> list:
        if self._done:
            return []
        return [0, 1]  # 0=check/fold, 1=bet/call

    def step(self, action: int):
        self.history.append(action)
        h = tuple(self.history)

        # Terminal conditions
        if h == (0, 0):  # check-check → showdown
            self._showdown()
        elif h == (0, 1, 0):  # check-bet-fold
            self._rewards = {0: -1.0, 1: 1.0}
            self._done = True
        elif h == (0, 1, 1):  # check-bet-call → showdown (pot=4)
            self._showdown(pot=4)
        elif h == (1, 0):  # bet-fold
            self._rewards = {0: 1.0, 1: -1.0}
            self._done = True
        elif h == (1, 1):  # bet-call → showdown (pot=4)
            self._showdown(pot=4)
        else:
            self.current_player = 1 - self.current_player

        return self._rewards, self._done

    def _showdown(self, pot: int = 2):
        half = pot / 2
        if self.cards[0] > self.cards[1]:
            self._rewards = {0: half, 1: -half}
        else:
            self._rewards = {0: -half, 1: half}
        self._done = True


class KuhnCFRSolver:
    """
    CFR solver specialized for Kuhn poker (for testing).

    Uses the same regret matching logic as the full solver,
    but with Kuhn's simpler game tree.
    """

    def __init__(self):
        self.info_sets: dict = {}  # key → InfoSet with 2 actions
        self.iterations = 0

    def _get_info_set(self, key: str) -> InfoSet:
        if key not in self.info_sets:
            self.info_sets[key] = InfoSet(num_actions=2)
        return self.info_sets[key]

    def _info_key(self, card: int, history: tuple) -> str:
        card_names = ["J", "Q", "K"]
        hist_str = "".join(str(a) for a in history)
        return f"{card_names[card]}|{hist_str}"

    def cfr(self, env: KuhnPokerEnv, traverser: int, history: tuple = ()) -> float:
        if env._done:
            return env._rewards.get(traverser, 0.0)

        current = env.current_player
        card = env.cards[current]
        key = self._info_key(card, history)
        info_set = self._get_info_set(key)
        strategy = info_set.current_strategy()

        if current == traverser:
            action_values = np.zeros(2, dtype=np.float64)
            node_value = 0.0

            for a in range(2):
                child = env.clone()
                child.step(a)
                action_values[a] = self.cfr(child, traverser, history + (a,))
                node_value += strategy[a] * action_values[a]

            for a in range(2):
                info_set.cumulative_regrets[a] += action_values[a] - node_value

            return node_value
        else:
            info_set.strategy_sum += strategy
            a = np.random.choice(2, p=strategy)
            child = env.clone()
            child.step(a)
            return self.cfr(child, traverser, history + (a,))

    def train(self, n_iterations: int):
        env = KuhnPokerEnv()
        for i in range(n_iterations):
            traverser = i % 2
            env.reset()
            self.cfr(env, traverser)
            self.iterations += 1
