"""
CFR → Live bridge
─────────────────
Produces Decision objects for the PokerStars autonomous agent.
Hot-swaps into autonomous_agent.py identically to the RL bridge.

Usage:
  python cfr_live_bridge.py --blueprint checkpoints/cfr/blueprint.npz \
      --ppo checkpoints/best.pt --abstraction checkpoints/cfr/abstraction.npz --dry-run
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional

from pokerStats.rl.poker_env import (
    PokerEnv, Action, NUM_ACTIONS, OBS_DIM,
    CARD_IDX, cards_to_onehot, STREETS,
    raise_frac_to_amount,
)
from pokerStats.rl.ppo_agent import PPOAgent
from .abstraction import HandAbstraction
from .blueprint import BlueprintStrategy
from .subgame_solver import SubgameSolver
from .hybrid_agent import HybridAgent


@dataclass
class Decision:
    """Output of the decision engine — matches reference/decision_engine.py."""
    action: str          # "fold" | "call" | "raise" | "check" | "allin"
    amount: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0


class CFRDecisionEngine:
    """
    Drop-in replacement for DecisionEngine / RLDecisionEngine.

    Uses the hybrid CFR+PPO agent to produce Decision objects.
    Same decide() signature as reference/decision_engine.py.
    """

    def __init__(
        self,
        blueprint_path: str,
        ppo_checkpoint_path: str,
        abstraction_path: str,
        exploit_blend: float = 0.3,
        use_subgame: bool = True,
        subgame_time: float = 2.0,
    ):
        # Load abstraction
        self.abstraction = HandAbstraction()
        self.abstraction.load(abstraction_path)

        # Load blueprint
        self.blueprint = BlueprintStrategy()
        self.blueprint.load(blueprint_path)

        # Load PPO agent
        self.ppo = PPOAgent()
        self.ppo.load(ppo_checkpoint_path)
        self.ppo.net.eval()

        # Build subgame solver
        subgame = None
        if use_subgame:
            subgame = SubgameSolver(
                self.abstraction, self.blueprint,
                time_limit=subgame_time,
            )

        # Build hybrid agent
        self.agent = HybridAgent(
            blueprint=self.blueprint,
            ppo_agent=self.ppo,
            abstraction=self.abstraction,
            subgame_solver=subgame,
            exploit_blend_base=exploit_blend,
        )

        print(f"CFR engine loaded: {len(self.blueprint)} info sets, "
              f"PPO ELO={self.ppo.elo:.0f}")

    def _build_obs(
        self,
        hole_cards: list,
        community_cards: list,
        street: str,
        pot: float,
        hero_stack: float,
        call_amount: float,
        position: str,
        num_opponents: int,
    ) -> np.ndarray:
        """Build observation vector for PPO component."""
        v = np.zeros(OBS_DIM, dtype=np.float32)
        v[0:52] = cards_to_onehot(hole_cards)
        v[52:104] = cards_to_onehot(community_cards)

        max_stack = 400.0
        v[104] = hero_stack / max_stack
        v[110] = call_amount / max_stack
        v[116] = 1.0  # hero active

        for i in range(min(num_opponents, 5)):
            v[117 + i] = 1.0

        pos_map = {"btn": 0, "sb": 1, "bb": 2, "utg": 3, "mp": 4, "co": 5, "hj": 5}
        v[122] = pos_map.get(str(position).lower(), 3) / 6.0

        v[128] = pot / max_stack
        v[129] = call_amount / max_stack
        v[130] = call_amount / max_stack
        v[131] = num_opponents / 5.0

        street_idx = STREETS.index(street) if street in STREETS else 0
        v[132 + street_idx] = 1.0

        return v

    def _legal_mask(self, available_actions: list, call_amount: float) -> np.ndarray:
        """Build env-space legal mask from string action names."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        action_map = {
            "fold": Action.FOLD,
            "check": Action.CHECK,
            "call": Action.CALL,
            "raise": Action.RAISE,
            "allin": Action.ALL_IN,
        }
        for a in available_actions:
            idx = action_map.get(a)
            if idx is not None:
                mask[int(idx)] = True

        if call_amount == 0 and not mask[Action.CHECK]:
            mask[Action.CHECK] = True
        if call_amount > 0:
            mask[Action.FOLD] = True

        return mask

    def _build_env_from_state(
        self,
        hole_cards: list,
        community_cards: list,
        street: str,
        pot: float,
        hero_stack: float,
        call_amount: float,
    ) -> PokerEnv:
        """
        Build a minimal PokerEnv representing the current live state.
        Used for blueprint lookup and subgame solving.
        """
        env = PokerEnv(num_players=2, render_mode="none")
        env.reset()

        # Override state to match live game
        env.players[0].hole_cards = list(hole_cards)
        env.players[0].stack = hero_stack
        env.community = list(community_cards)
        env.pot = pot
        env.street_idx = STREETS.index(street) if street in STREETS else 0
        env.current_player = 0
        env.last_bet = call_amount if call_amount > 0 else env.big_blind
        env._done = False

        return env

    def decide(
        self,
        hole_cards: list,
        community_cards: list,
        street: str,
        pot: float,
        hero_stack: float,
        call_amount: float,
        position: str,
        num_opponents: int,
        opponent_profiles: dict,
        available_actions: list,
    ) -> Decision:
        """
        Produce a Decision using the hybrid CFR+PPO agent.

        Same signature as DecisionEngine.decide() and RLDecisionEngine.decide().
        """
        if not hole_cards:
            return Decision("fold", reasoning="no cards detected")

        # Build env for CFR
        env = self._build_env_from_state(
            hole_cards, community_cards, street,
            pot, hero_stack, call_amount,
        )

        # Build obs for PPO
        obs = self._build_obs(
            hole_cards, community_cards, street,
            pot, hero_stack, call_amount,
            position, num_opponents,
        )
        legal_mask = self._legal_mask(available_actions, call_amount)

        # Opponent confidence from profiles
        profiles = list(opponent_profiles.values()) if opponent_profiles else []
        total_hands = sum(getattr(p, "hands_played", 0) for p in profiles)
        avg_hands = total_hands / max(len(profiles), 1)

        # Get hybrid action
        action_int, raise_frac, source = self.agent.get_action(
            env=env,
            obs=obs,
            legal_mask_env=legal_mask,
            action_history=(),
            opponent_hands_observed=int(avg_hands),
        )

        # Convert to Decision
        action = Action(action_int)
        if action == Action.FOLD:
            return Decision("fold", 0.0, f"CFR {source}", confidence=0.9)
        elif action == Action.CHECK:
            return Decision("check", 0.0, f"CFR {source}", confidence=0.9)
        elif action == Action.CALL:
            return Decision("call", call_amount, f"CFR {source}", confidence=0.9)
        elif action == Action.RAISE:
            min_raise = max(call_amount * 2, env.big_blind * 2)
            max_raise = hero_stack
            amount = raise_frac_to_amount(raise_frac, min_raise, max_raise)
            amount = min(amount, hero_stack)
            return Decision("raise", round(amount, 2), f"CFR {source}", confidence=0.85)
        elif action == Action.ALL_IN:
            return Decision("allin", hero_stack, f"CFR {source}", confidence=0.9)

        return Decision("check", 0.0, f"CFR fallback", confidence=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFR+PPO live bridge")
    parser.add_argument("--blueprint", required=True, help="Path to blueprint .npz")
    parser.add_argument("--ppo", required=True, help="Path to PPO checkpoint .pt")
    parser.add_argument("--abstraction", required=True, help="Path to abstraction .npz")
    parser.add_argument("--exploit-blend", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = CFRDecisionEngine(
        blueprint_path=args.blueprint,
        ppo_checkpoint_path=args.ppo,
        abstraction_path=args.abstraction,
        exploit_blend=args.exploit_blend,
    )

    # Hot-swap into autonomous agent
    try:
        from reference.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(dry_run=args.dry_run)
        agent.engine = engine
        print(f"\nCFR+PPO agent active  dry_run={args.dry_run}")
        agent.run()
    except ImportError:
        print("autonomous_agent not available — running in standalone mode")
        # Demo decision
        decision = engine.decide(
            hole_cards=["Ah", "Ks"],
            community_cards=["Qh", "Jh", "2c"],
            street="flop",
            pot=50.0,
            hero_stack=200.0,
            call_amount=10.0,
            position="btn",
            num_opponents=1,
            opponent_profiles={},
            available_actions=["fold", "call", "raise"],
        )
        print(f"Decision: {decision}")
