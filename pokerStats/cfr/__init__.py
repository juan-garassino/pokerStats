"""
CFR (Counterfactual Regret Minimization) subpackage
────────────────────────────────────────────────────
Hand abstraction, MCCFR solver, blueprint strategy,
real-time subgame solving, and hybrid CFR+PPO agent.
"""

from .abstraction import PreflopAbstraction, PostflopAbstraction, HandAbstraction
from .cfr_solver import CFRSolver, InfoSet
from .blueprint import BlueprintStrategy
from .subgame_solver import SubgameSolver
from .hybrid_agent import HybridAgent
from .cfr_live_bridge import CFRDecisionEngine

__all__ = [
    "PreflopAbstraction", "PostflopAbstraction", "HandAbstraction",
    "CFRSolver", "InfoSet",
    "BlueprintStrategy",
    "SubgameSolver",
    "HybridAgent",
    "CFRDecisionEngine",
]
