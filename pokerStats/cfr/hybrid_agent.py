"""
Hybrid CFR + PPO agent
──────────────────────
Blends a GTO blueprint (CFR) with an exploitation policy (PPO).

    final_policy = (1 - alpha) * blueprint_policy + alpha * ppo_exploit_policy

- alpha = exploit_blend_base × opponent_confidence
- opponent_confidence ramps from 0 (no data) to 1 (30+ hands observed)
- Big pots (>30 BB): optionally runs SubgameSolver to refine blueprint
- Falls back to pure PPO if blueprint has no coverage
"""

import numpy as np
from typing import Optional

from pokerStats.rl.poker_env import (
    PokerEnv, Action, NUM_ACTIONS, OBS_DIM, STREETS,
    hand_strength, raise_frac_to_amount,
)
from pokerStats.rl.ppo_agent import PPOAgent
from .abstraction import HandAbstraction
from .blueprint import BlueprintStrategy
from .subgame_solver import SubgameSolver
from .cfr_solver import NUM_CFR_ACTIONS, CFR_ACTIONS, cfr_to_env_action, env_legal_to_cfr_mask


class HybridAgent:
    """
    Combines CFR blueprint (GTO floor) with PPO policy (exploitation ceiling).

    Decision flow:
    1. Look up blueprint strategy for current (hand_bucket, street, action_history)
    2. Get PPO policy from PPOAgent.get_action()
    3. Compute blend weight from opponent model confidence
    4. Big pots → subgame solve (replaces blueprint component)
    5. Blend, mask illegal actions, sample action
    6. Use PPO's Beta distribution for raise sizing
    """

    def __init__(
        self,
        blueprint: BlueprintStrategy,
        ppo_agent: PPOAgent,
        abstraction: HandAbstraction,
        subgame_solver: Optional[SubgameSolver] = None,
        exploit_blend_base: float = 0.3,
        big_pot_threshold_bb: float = 30.0,
        confidence_hands: int = 30,
    ):
        self.blueprint = blueprint
        self.ppo = ppo_agent
        self.abstraction = abstraction
        self.subgame = subgame_solver
        self.exploit_blend_base = exploit_blend_base
        self.big_pot_threshold_bb = big_pot_threshold_bb
        self.confidence_hands = confidence_hands

    def get_action(
        self,
        env: PokerEnv,
        obs: np.ndarray,
        legal_mask_env: np.ndarray,
        action_history: tuple = (),
        opponent_hands_observed: int = 0,
        use_subgame: bool = True,
        deterministic: bool = False,
    ) -> tuple:
        """
        Get blended CFR+PPO action.

        Args:
            env: Current game state.
            obs: Observation vector for current player (OBS_DIM,).
            legal_mask_env: Legal mask in env action space (NUM_ACTIONS,).
            action_history: CFR action indices taken this hand.
            opponent_hands_observed: Number of hands observed on primary opponent.
            use_subgame: Whether to use subgame solving for big pots.
            deterministic: If True, take argmax instead of sampling.

        Returns:
            (action: int, raise_frac: float, source: str)
            action is in env Action space, raise_frac in [0, 1].
        """
        player = env.players[env.current_player]
        hand_bucket = self.abstraction.get_bucket(
            player.hole_cards, env.community
        )
        street = STREETS[env.street_idx]

        # ── 1. Blueprint lookup ──────────────────────────────────────────
        bp_policy = self.blueprint.lookup(hand_bucket, street, action_history)

        # ── 2. Big pot subgame solving ───────────────────────────────────
        pot_bb = env.pot / env.big_blind
        if (bp_policy is not None and use_subgame and
                self.subgame is not None and
                pot_bb > self.big_pot_threshold_bb):
            refined = self.subgame.solve(env, env.current_player)
            if refined is not None:
                bp_policy = refined

        # ── 3. Get PPO policy ────────────────────────────────────────────
        ppo_action, ppo_raise_frac, _, _, _ = self.ppo.get_action(
            obs, legal_mask_env, deterministic=deterministic
        )

        # Build PPO probability distribution over CFR action space
        ppo_probs_cfr = self._ppo_action_to_cfr_probs(ppo_action, legal_mask_env)

        # ── 4. Compute blend weight ─────────────────────────────────────
        opponent_confidence = min(1.0, opponent_hands_observed / self.confidence_hands)
        alpha = self.exploit_blend_base * opponent_confidence

        # ── 5. Blend strategies ──────────────────────────────────────────
        cfr_legal_mask = env_legal_to_cfr_mask(env)

        if bp_policy is not None:
            # Blend: (1-alpha)*blueprint + alpha*ppo
            blended = (1.0 - alpha) * bp_policy + alpha * ppo_probs_cfr
            source = f"hybrid(alpha={alpha:.2f})"
        else:
            # No blueprint coverage: pure PPO
            blended = ppo_probs_cfr
            source = "ppo_fallback"

        # Mask illegal and renormalize
        blended *= cfr_legal_mask
        total = blended.sum()
        if total > 0:
            blended /= total
        else:
            # Emergency: uniform over legal
            blended = cfr_legal_mask.astype(np.float64)
            blended /= blended.sum()

        # ── 6. Sample action ─────────────────────────────────────────────
        if deterministic:
            cfr_action = int(np.argmax(blended))
        else:
            cfr_action = int(np.random.choice(NUM_CFR_ACTIONS, p=blended))

        env_action, cfr_raise_frac = cfr_to_env_action(cfr_action)

        # Use PPO's Beta distribution for raise sizing (more nuanced than CFR's discrete sizes)
        if env_action == Action.RAISE:
            raise_frac = ppo_raise_frac
        else:
            raise_frac = cfr_raise_frac

        return int(env_action), float(raise_frac), source

    def _ppo_action_to_cfr_probs(
        self,
        ppo_action: int,
        legal_mask_env: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a PPO discrete action to a probability distribution
        over the CFR action space.

        Maps:
            FOLD → cfr[0]
            CHECK → cfr[1]
            CALL → cfr[2]
            RAISE → split across cfr[3,4,5] (33%, 75%, 150% pot)
            ALL_IN → cfr[6]
        """
        probs = np.zeros(NUM_CFR_ACTIONS, dtype=np.float64)

        if ppo_action == Action.FOLD:
            probs[0] = 1.0
        elif ppo_action == Action.CHECK:
            probs[1] = 1.0
        elif ppo_action == Action.CALL:
            probs[2] = 1.0
        elif ppo_action == Action.RAISE:
            # Spread across raise sizes
            probs[3] = 0.33
            probs[4] = 0.34
            probs[5] = 0.33
        elif ppo_action == Action.ALL_IN:
            probs[6] = 1.0

        return probs

    def get_action_pure_blueprint(
        self,
        env: PokerEnv,
        action_history: tuple = (),
        deterministic: bool = False,
    ) -> tuple:
        """Get action using pure blueprint (alpha=0, no PPO blending)."""
        player = env.players[env.current_player]
        hand_bucket = self.abstraction.get_bucket(
            player.hole_cards, env.community
        )
        street = STREETS[env.street_idx]
        bp_policy = self.blueprint.lookup(hand_bucket, street, action_history)
        cfr_legal_mask = env_legal_to_cfr_mask(env)

        if bp_policy is None:
            # No coverage: uniform over legal
            bp_policy = cfr_legal_mask.astype(np.float64)
            total = bp_policy.sum()
            if total > 0:
                bp_policy /= total

        bp_policy *= cfr_legal_mask
        total = bp_policy.sum()
        if total > 0:
            bp_policy /= total
        else:
            bp_policy = cfr_legal_mask.astype(np.float64) / max(cfr_legal_mask.sum(), 1)

        if deterministic:
            cfr_action = int(np.argmax(bp_policy))
        else:
            cfr_action = int(np.random.choice(NUM_CFR_ACTIONS, p=bp_policy))

        env_action, raise_frac = cfr_to_env_action(cfr_action)
        return int(env_action), float(raise_frac), "blueprint"
