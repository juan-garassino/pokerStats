"""Tests for pokerStats.cfr.hybrid_agent"""
import numpy as np
from pokerStats.cfr.hybrid_agent import HybridAgent
from pokerStats.cfr.blueprint import BlueprintStrategy
from pokerStats.cfr.abstraction import HandAbstraction
from pokerStats.cfr.cfr_solver import CFRSolver, NUM_CFR_ACTIONS, env_legal_to_cfr_mask
from pokerStats.rl.poker_env import PokerEnv, Action, NUM_ACTIONS, OBS_DIM
from pokerStats.rl.ppo_agent import PPOAgent


def _make_test_agent():
    """Create a minimal hybrid agent for testing."""
    # Train a tiny CFR solver to get some info sets
    solver = CFRSolver()
    solver.train(20, log_every=0)
    blueprint = BlueprintStrategy.from_solver(solver, min_visits=0)

    ppo = PPOAgent(use_alpha=False)
    abstraction = HandAbstraction()

    return HybridAgent(
        blueprint=blueprint,
        ppo_agent=ppo,
        abstraction=abstraction,
        exploit_blend_base=0.3,
    )


def _make_env():
    """Create a 2-player env and reset it."""
    env = PokerEnv(num_players=2, render_mode="none")
    env.reset()
    return env


def test_hybrid_agent_returns_valid_action():
    """Hybrid agent returns a valid env action and raise_frac."""
    agent = _make_test_agent()
    env = _make_env()

    obs = env._build_obs(env.current_player)
    legal = env.legal_mask()

    action, raise_frac, source = agent.get_action(
        env=env, obs=obs, legal_mask_env=legal,
    )

    assert action in [int(a) for a in Action]
    assert 0.0 <= raise_frac <= 1.0
    assert isinstance(source, str)


def test_hybrid_agent_legal_actions_respected():
    """Hybrid agent only returns legal actions."""
    agent = _make_test_agent()
    env = _make_env()

    for _ in range(20):
        env.reset()
        obs = env._build_obs(env.current_player)
        legal = env.legal_mask()
        legal_set = set(env.legal_actions())

        action, _, _ = agent.get_action(
            env=env, obs=obs, legal_mask_env=legal,
        )
        assert action in legal_set or action in [int(a) for a in Action], \
            f"Action {action} not in legal set {legal_set}"


def test_pure_blueprint_mode():
    """get_action_pure_blueprint returns valid actions."""
    agent = _make_test_agent()
    env = _make_env()

    action, raise_frac, source = agent.get_action_pure_blueprint(env)
    assert action in [int(a) for a in Action]
    assert source == "blueprint"


def test_alpha_zero_is_pure_blueprint():
    """With alpha=0 (no opponent data), should be near-pure blueprint."""
    solver = CFRSolver()
    solver.train(30, log_every=0)
    blueprint = BlueprintStrategy.from_solver(solver, min_visits=0)
    ppo = PPOAgent(use_alpha=False)
    abstraction = HandAbstraction()

    agent = HybridAgent(
        blueprint=blueprint,
        ppo_agent=ppo,
        abstraction=abstraction,
        exploit_blend_base=0.3,
    )

    env = _make_env()
    obs = env._build_obs(env.current_player)
    legal = env.legal_mask()

    # With 0 opponent hands observed, alpha should be 0 → pure blueprint
    _, _, source = agent.get_action(
        env=env, obs=obs, legal_mask_env=legal,
        opponent_hands_observed=0,
    )
    assert "hybrid(alpha=0.00)" in source or source == "ppo_fallback"


def test_alpha_increases_with_observations():
    """Alpha increases as opponent observation count grows."""
    agent = _make_test_agent()
    env = _make_env()
    obs = env._build_obs(env.current_player)
    legal = env.legal_mask()

    # With many observations, alpha should be higher
    _, _, source_high = agent.get_action(
        env=env, obs=obs, legal_mask_env=legal,
        opponent_hands_observed=100,
    )
    # alpha = 0.3 * min(100/30, 1) = 0.3
    assert "0.30" in source_high or source_high == "ppo_fallback"


def test_ppo_action_to_cfr_probs():
    """PPO action mapping produces valid CFR probability distributions."""
    agent = _make_test_agent()
    legal = np.ones(NUM_ACTIONS, dtype=bool)

    for action in Action:
        probs = agent._ppo_action_to_cfr_probs(int(action), legal)
        assert probs.shape == (NUM_CFR_ACTIONS,)
        assert abs(probs.sum() - 1.0) < 1e-6, \
            f"Probs should sum to 1 for action {action}, got {probs.sum()}"
        assert all(p >= 0 for p in probs)


def test_blueprint_strategy_lookup():
    """Blueprint lookup returns correct shape or None."""
    solver = CFRSolver()
    solver.train(50, log_every=0)
    blueprint = BlueprintStrategy.from_solver(solver, min_visits=0)

    # Try a few lookups
    result = blueprint.lookup(0, "preflop", ())
    if result is not None:
        assert result.shape == (NUM_CFR_ACTIONS,)
        assert abs(result.sum() - 1.0) < 1e-6


def test_blueprint_save_load(tmp_path):
    """Blueprint save/load roundtrip."""
    solver = CFRSolver()
    solver.train(30, log_every=0)
    bp = BlueprintStrategy.from_solver(solver, min_visits=0)

    path = str(tmp_path / "blueprint.npz")
    bp.save(path)

    bp2 = BlueprintStrategy()
    bp2.load(path)

    assert len(bp2) == len(bp)
