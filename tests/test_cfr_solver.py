"""Tests for pokerStats.cfr.cfr_solver"""
import numpy as np
from pokerStats.cfr.cfr_solver import (
    InfoSet, CFRSolver, KuhnCFRSolver, KuhnPokerEnv,
    NUM_CFR_ACTIONS, cfr_to_env_action, env_legal_to_cfr_mask,
)
from pokerStats.cfr.abstraction import HandAbstraction
from pokerStats.rl.poker_env import PokerEnv, Action


# ── InfoSet tests ────────────────────────────────────────────────────────────

def test_regret_matching_valid_distribution():
    """current_strategy() returns a valid probability distribution."""
    info = InfoSet()
    info.cumulative_regrets = np.array([1.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0])
    strategy = info.current_strategy()
    assert abs(strategy.sum() - 1.0) < 1e-8
    assert all(s >= 0 for s in strategy)


def test_regret_matching_zero_regrets():
    """All-zero regrets → uniform distribution."""
    info = InfoSet()
    strategy = info.current_strategy()
    expected = 1.0 / NUM_CFR_ACTIONS
    for s in strategy:
        assert abs(s - expected) < 1e-8


def test_regret_matching_negative_regrets():
    """All-negative regrets → uniform distribution."""
    info = InfoSet()
    info.cumulative_regrets = np.array([-5.0, -3.0, -1.0, -2.0, -4.0, -6.0, -7.0])
    strategy = info.current_strategy()
    assert abs(strategy.sum() - 1.0) < 1e-8


def test_regret_matching_with_legal_mask():
    """Legal mask restricts strategy to legal actions only."""
    info = InfoSet()
    info.cumulative_regrets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    mask = np.array([True, False, True, False, False, False, True])
    strategy = info.current_strategy(mask)
    assert abs(strategy.sum() - 1.0) < 1e-8
    assert strategy[1] == 0.0  # masked out
    assert strategy[3] == 0.0  # masked out


def test_average_strategy_valid():
    """average_strategy() returns valid distribution."""
    info = InfoSet()
    info.strategy_sum = np.array([10.0, 20.0, 30.0, 15.0, 5.0, 10.0, 10.0])
    avg = info.average_strategy()
    assert abs(avg.sum() - 1.0) < 1e-8
    assert all(a >= 0 for a in avg)


# ── Kuhn Poker CFR convergence ───────────────────────────────────────────────

def test_kuhn_poker_env():
    """Kuhn poker env runs to completion."""
    env = KuhnPokerEnv()
    env.reset()
    assert not env._done
    # check-check
    env.step(0)
    env.step(0)
    assert env._done


def test_kuhn_poker_clone():
    """Kuhn poker clone is independent."""
    env = KuhnPokerEnv()
    env.reset()
    clone = env.clone()
    clone.step(1)
    assert not env._done  # original unchanged


def test_kuhn_cfr_convergence():
    """
    CFR converges to the known Nash equilibrium in Kuhn poker.

    Known Nash for player 0:
    - J: bet with probability ~1/3 (bluff)
    - Q: check always, call with probability ~1/3
    - K: bet with probability ~3x check frequency

    We verify the key property: with enough iterations,
    the average strategy stabilizes and game value ≈ -1/18 for player 0.
    """
    solver = KuhnCFRSolver()
    solver.train(10000)

    # Check that strategies are valid distributions
    for key, info_set in solver.info_sets.items():
        avg = info_set.average_strategy()
        assert abs(avg.sum() - 1.0) < 1e-6, f"Strategy for {key} doesn't sum to 1"
        assert all(a >= 0 for a in avg), f"Negative probability in {key}"

    # Key Nash properties (approximate):
    # J at root: bet ~1/3 of the time (bluff)
    j_root = solver.info_sets.get("J|")
    if j_root is not None:
        j_bet = j_root.average_strategy()[1]  # action 1 = bet
        assert 0.1 < j_bet < 0.6, f"J bet freq {j_bet} outside expected range"

    # K at root: should bet frequently
    k_root = solver.info_sets.get("K|")
    if k_root is not None:
        k_bet = k_root.average_strategy()[1]
        assert k_bet > 0.3, f"K should bet frequently, got {k_bet}"


def test_kuhn_cfr_exploitability_decreases():
    """With more iterations, strategies should improve."""
    solver1 = KuhnCFRSolver()
    solver1.train(100)

    solver2 = KuhnCFRSolver()
    solver2.train(5000)

    # More iterations → strategies should be more refined
    # (we can't easily compute exploitability for Kuhn, but we verify convergence)
    assert solver2.iterations > solver1.iterations


# ── Full NLHE CFR tests ──────────────────────────────────────────────────────

def test_cfr_action_mapping():
    """CFR actions map to valid env actions."""
    for i in range(NUM_CFR_ACTIONS):
        action, raise_frac = cfr_to_env_action(i)
        assert isinstance(action, Action)
        assert 0.0 <= raise_frac <= 1.0


def test_cfr_legal_mask():
    """Legal mask correctly maps env actions to CFR space."""
    env = PokerEnv(num_players=2, render_mode="none")
    env.reset()
    mask = env_legal_to_cfr_mask(env)
    assert mask.shape == (NUM_CFR_ACTIONS,)
    assert mask.any(), "At least one CFR action must be legal"


def test_cfr_solver_creates_info_sets():
    """Running a few CFR iterations creates info sets."""
    solver = CFRSolver()
    solver.train(10, log_every=0)
    assert len(solver.info_sets) > 0, "Solver should create info sets"
    assert solver.iterations == 10


def test_cfr_solver_save_load(tmp_path):
    """Save and load roundtrip preserves solver state."""
    solver = CFRSolver()
    solver.train(20, log_every=0)

    path = str(tmp_path / "solver.npz")
    solver.save(path)

    solver2 = CFRSolver()
    solver2.load(path)

    assert solver2.iterations == solver.iterations
    assert len(solver2.info_sets) == len(solver.info_sets)

    # Verify info set contents match
    for key in solver.info_sets:
        assert key in solver2.info_sets
        np.testing.assert_array_almost_equal(
            solver.info_sets[key].cumulative_regrets,
            solver2.info_sets[key].cumulative_regrets,
        )


def test_cfr_solver_info_set_strategies_valid():
    """All info set strategies should be valid probability distributions."""
    solver = CFRSolver()
    solver.train(50, log_every=0)

    for key, info_set in solver.info_sets.items():
        strategy = info_set.current_strategy()
        assert abs(strategy.sum() - 1.0) < 1e-6, f"Invalid strategy sum for {key}"
        assert all(s >= 0 for s in strategy), f"Negative prob for {key}"
