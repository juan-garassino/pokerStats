"""
Information Set Monte Carlo Tree Search for poker
───────────────────────────────────────────────────
Uses the policy network as prior and value network as leaf evaluation.
Handles imperfect information by sampling opponent hands from the range head.

Only used at inference time — training uses raw policy for speed.
"""

import math
import numpy as np
import torch
from typing import Optional

from .poker_env import PokerEnv, Action, NUM_ACTIONS, ALL_CARDS, CARD_IDX


class MCTSNode:
    """A node in the search tree."""
    __slots__ = ('parent', 'action', 'prior', 'visit_count', 'value_sum',
                 'children', 'is_terminal')

    def __init__(self, parent=None, action: int = -1, prior: float = 0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_terminal = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float = 1.5) -> float:
        if self.parent is None:
            return 0.0
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / \
                      (1 + self.visit_count)
        return self.q_value + exploration

    def best_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))

    def expand(self, action_priors: dict):
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, action=action,
                                                  prior=prior)

    def backup(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip for opponent
            node = node.parent


class PokerMCTS:
    """
    Monte Carlo Tree Search for poker.
    Uses IS-MCTS: sample opponent hands from range head, then search
    the determinized game tree.

    Args:
        net: AlphaPokerNet (or any net with .forward() returning policy + value)
        num_simulations: number of MCTS rollouts per decision
        c_puct: exploration constant for UCB
        max_depth: max search depth (streets beyond current)
    """

    def __init__(self, net, num_simulations: int = 100, c_puct: float = 1.5,
                 max_depth: int = 2):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth

    def _sample_opponent_hands(self, range_pred: np.ndarray,
                                hero_cards: list,
                                community: list,
                                num_opponents: int) -> list:
        """
        Sample plausible opponent hands from range head prediction.
        Returns list of (card1, card2) tuples for each opponent.
        """
        # Cards already in use
        used = set()
        for c in hero_cards + community:
            if c in CARD_IDX:
                used.add(CARD_IDX[c])

        available = [i for i in range(52) if i not in used]
        opponent_hands = []

        for opp_i in range(num_opponents):
            # Get this opponent's card probabilities
            start = opp_i * 52
            end = start + 52
            if end <= len(range_pred):
                probs = range_pred[start:end].copy()
            else:
                probs = np.ones(52) / 52.0

            # Zero out used cards
            for idx in used:
                probs[idx] = 0.0

            # Filter to available cards
            total = probs.sum()
            if total < 1e-8:
                probs = np.ones(52)
                for idx in used:
                    probs[idx] = 0.0
                total = probs.sum()

            probs = probs / total

            # Sample 2 cards without replacement
            avail_probs = probs.copy()
            cards = []
            for _ in range(2):
                avail_total = avail_probs.sum()
                if avail_total < 1e-8:
                    # Fallback: uniform over remaining
                    remaining = [i for i in range(52) if i not in used and i not in cards]
                    if remaining:
                        card = np.random.choice(remaining)
                    else:
                        break
                else:
                    card = np.random.choice(52, p=avail_probs / avail_total)

                cards.append(card)
                avail_probs[card] = 0.0
                used.add(card)

            if len(cards) == 2:
                opponent_hands.append((ALL_CARDS[cards[0]], ALL_CARDS[cards[1]]))
            else:
                opponent_hands.append(None)

        return opponent_hands

    def _determinize(self, env: PokerEnv, opp_hands: list,
                     hero_idx: int) -> PokerEnv:
        """Create a determinized copy of the env with sampled opponent hands."""
        sim_env = env.clone()
        opp_slot = 0
        for pidx in range(sim_env.num_players):
            if pidx == hero_idx:
                continue
            if opp_slot < len(opp_hands) and opp_hands[opp_slot] is not None:
                sim_env.players[pidx].hole_cards = list(opp_hands[opp_slot])
            opp_slot += 1
        return sim_env

    @torch.inference_mode()
    def _evaluate(self, env: PokerEnv, hero_idx: int,
                  obs: np.ndarray, legal_mask: np.ndarray,
                  opp_events: Optional[np.ndarray] = None,
                  opp_masks: Optional[np.ndarray] = None) -> tuple:
        """Get policy prior and value from the network."""
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        mask_t = torch.from_numpy(legal_mask.astype(np.bool_)).unsqueeze(0)

        # Handle both AlphaPokerNet (7 returns) and legacy PokerNet (4 returns)
        is_alpha = hasattr(self.net, 'opponent_encoder')

        if is_alpha and opp_events is not None:
            opp_ev_t = torch.from_numpy(opp_events).unsqueeze(0)
            opp_mk_t = torch.from_numpy(opp_masks).unsqueeze(0)
            logits, _, _, value, _, _, range_pred = self.net(
                obs_t, mask_t, opp_ev_t, opp_mk_t
            )
            range_np = range_pred[0].numpy()
        elif is_alpha:
            logits, _, _, value, _, _, range_pred = self.net(obs_t, mask_t)
            range_np = range_pred[0].numpy()
        else:
            logits, _, _, value = self.net(obs_t, mask_t)
            range_np = np.ones(52) / 52  # uniform range for legacy net

        # Convert logits to probabilities
        logits_np = logits[0].numpy()
        logits_np = logits_np - logits_np.max()
        probs = np.exp(logits_np)
        probs = probs / (probs.sum() + 1e-8)

        legal = [i for i in range(NUM_ACTIONS) if legal_mask[i]]
        action_priors = {a: float(probs[a]) for a in legal}

        return action_priors, float(value[0].numpy()), range_np

    def _rollout_value(self, env: PokerEnv, hero_idx: int, depth: int) -> float:
        """Quick rollout using random actions to estimate value."""
        if env._done:
            return env._rewards.get(hero_idx, 0.0)

        if depth >= self.max_depth * 10:  # emergency cutoff
            return 0.0

        legal = env.legal_actions()
        if not legal:
            return 0.0

        action = np.random.choice(legal)
        _, rewards, done, _ = env.step(action)

        if done:
            return rewards.get(hero_idx, 0.0)

        return self._rollout_value(env, hero_idx, depth + 1)

    def search(self, env: PokerEnv, hero_idx: int,
               obs: np.ndarray, legal_mask: np.ndarray,
               opp_events: Optional[np.ndarray] = None,
               opp_masks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run IS-MCTS from the current state.

        Returns: action probability distribution (NUM_ACTIONS,) based on visit counts.
        """
        # Get initial evaluation
        action_priors, root_value, range_pred = self._evaluate(
            env, hero_idx, obs, legal_mask, opp_events, opp_masks
        )

        root = MCTSNode()
        root.expand(action_priors)

        hero_cards = env.players[hero_idx].hole_cards
        community = env.community
        num_opp = env.num_players - 1

        for _ in range(self.num_simulations):
            # Sample opponent hands from range prediction
            opp_hands = self._sample_opponent_hands(
                range_pred, hero_cards, community, num_opp
            )

            # Create determinized env
            sim_env = self._determinize(env, opp_hands, hero_idx)

            # Select: walk down tree using UCB
            node = root
            while node.children and not node.is_terminal:
                node = node.best_child(self.c_puct)

                # Apply action in sim env
                if not sim_env._done:
                    raise_frac = 0.3  # default raise sizing for search
                    _, _, done, _ = sim_env.step(node.action, raise_frac)
                    if done:
                        node.is_terminal = True
                        break

            # Expand if not terminal
            if not node.is_terminal and not sim_env._done:
                sim_legal = sim_env.legal_mask()
                sim_obs = sim_env._build_obs(sim_env.current_player)
                child_priors, leaf_value, _ = self._evaluate(
                    sim_env, hero_idx, sim_obs, sim_legal
                )
                node.expand(child_priors)
                value = leaf_value
            elif sim_env._done:
                value = sim_env._rewards.get(hero_idx, 0.0)
            else:
                value = 0.0

            # Backpropagate
            node.backup(value)

        # Extract action probabilities from visit counts
        action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits > 0:
            for action, child in root.children.items():
                action_probs[action] = child.visit_count / total_visits

        return action_probs

    def get_action(self, env: PokerEnv, hero_idx: int,
                   obs: np.ndarray, legal_mask: np.ndarray,
                   opp_events: Optional[np.ndarray] = None,
                   opp_masks: Optional[np.ndarray] = None,
                   temperature: float = 0.1) -> int:
        """
        Run MCTS and return the best action.
        Low temperature → greedy, high temperature → exploratory.
        """
        action_probs = self.search(env, hero_idx, obs, legal_mask,
                                    opp_events, opp_masks)

        if temperature < 1e-3:
            return int(np.argmax(action_probs))

        # Apply temperature
        log_probs = np.log(action_probs + 1e-8) / temperature
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

        return int(np.random.choice(NUM_ACTIONS, p=probs))
