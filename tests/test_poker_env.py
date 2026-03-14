"""Tests for pokerStats.rl.poker_env"""
import numpy as np
from pokerStats.rl.poker_env import (
    PokerEnv, Action, NUM_ACTIONS, OBS_DIM,
    hand_strength, draw_potential, raise_frac_to_amount,
)


def test_obs_shape():
    """Observation vectors have the correct dimension."""
    env = PokerEnv(num_players=6)
    obs = env.reset()
    assert OBS_DIM == 148
    for i in range(6):
        assert obs[i].shape == (OBS_DIM,), f"Player {i} obs shape mismatch"


def test_legal_actions_nonempty():
    """Current player always has at least one legal action."""
    env = PokerEnv(num_players=6)
    env.reset()
    legal = env.legal_actions()
    assert len(legal) > 0, "Legal actions should not be empty after reset"


def test_legal_mask_shape():
    """Legal mask has correct shape and at least one True."""
    env = PokerEnv(num_players=6)
    env.reset()
    mask = env.legal_mask()
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.any(), "At least one action must be legal"


def test_hand_completes():
    """A hand eventually terminates when all players fold or call."""
    env = PokerEnv(num_players=3)
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        legal = env.legal_actions()
        # Pick first legal action (often fold/check)
        action = legal[0]
        obs, rewards, done, info = env.step(action)
        steps += 1
    assert done, f"Hand did not complete in {steps} steps"
    assert isinstance(rewards, dict)
    assert 0 in rewards


def test_action_ids_populated():
    """_action_ids is populated after stepping (bug fix verification)."""
    env = PokerEnv(num_players=3)
    env.reset()
    assert len(env._action_ids) == 0, "Should start empty"

    legal = env.legal_actions()
    env.step(legal[0])

    assert len(env._action_ids) >= 1, "_action_ids should be populated after step"


def test_rewards_sum_to_zero():
    """Total rewards across all players should be approximately zero (zero-sum)."""
    env = PokerEnv(num_players=4)
    obs = env.reset()
    done = False
    while not done:
        legal = env.legal_actions()
        obs, rewards, done, _ = env.step(legal[0])

    total = sum(rewards.values())
    assert abs(total) < 0.01, f"Rewards should sum to ~0, got {total}"


def test_multiple_hands():
    """Can play multiple hands in sequence."""
    env = PokerEnv(num_players=3)
    for _ in range(5):
        obs = env.reset()
        done = False
        while not done:
            legal = env.legal_actions()
            obs, rewards, done, _ = env.step(legal[0])
    assert env.hand_num == 5


def test_num_actions():
    """Action space has 5 discrete actions + continuous raise sizing."""
    assert NUM_ACTIONS == 5


def test_hand_strength_preflop():
    """Hand strength heuristic works preflop (no community)."""
    # Pocket aces should be near 1.0
    aa = hand_strength(["Ah", "As"], [])
    assert aa > 0.9

    # 2-7 offsuit should be low
    low = hand_strength(["2c", "7d"], [])
    assert low < 0.3

    assert aa > low


def test_hand_strength_postflop():
    """Hand strength uses treys postflop when available."""
    hs = hand_strength(["Ah", "Kh"], ["Qh", "Jh", "Th"])
    # Royal flush — should be very high
    assert hs > 0.95


def test_draw_potential():
    """Flush and straight draws detected."""
    # 4 hearts = flush draw
    dp = draw_potential(["Ah", "2h"], ["5h", "8h", "Tc"])
    assert dp >= 0.5

    # No draw
    dp_none = draw_potential(["Ac", "Kd"], ["2h", "7s", "Js"])
    assert dp_none < 0.5


def test_obs_has_hand_strength():
    """Obs vector slots 143-144 contain hand strength and draw potential."""
    env = PokerEnv(num_players=3)
    obs = env.reset()
    for i in range(3):
        # Hand strength should be between 0 and 1
        assert 0.0 <= obs[i][143] <= 1.0
        assert 0.0 <= obs[i][144] <= 1.0


def test_raise_curve_boundaries():
    """Exponential raise curve maps 0->min and 1->max."""
    result_0 = raise_frac_to_amount(0.0, 10.0, 200.0)
    result_1 = raise_frac_to_amount(1.0, 10.0, 200.0)
    assert abs(result_0 - 10.0) < 0.01, f"frac=0 should give min_raise, got {result_0}"
    assert abs(result_1 - 200.0) < 0.01, f"frac=1 should give max_raise, got {result_1}"


def test_raise_curve_is_exponential():
    """Small fracs give proportionally smaller raises (exponential shape)."""
    # At frac=0.5, result should be less than midpoint (exponential curve)
    mid = raise_frac_to_amount(0.5, 0.0, 100.0)
    assert mid < 50.0, f"Exponential curve should give <50 at frac=0.5, got {mid}"


def test_continuous_raise_in_step():
    """step() accepts raise_frac and produces valid raise amounts."""
    env = PokerEnv(num_players=3)
    env.reset()
    # Find a state where raise is legal
    legal = env.legal_actions()
    if Action.RAISE in legal:
        obs, rewards, done, info = env.step(Action.RAISE, 0.3)
        assert isinstance(rewards, dict)
