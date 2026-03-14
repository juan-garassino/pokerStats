"""Tests for pokerStats.rl.poker_env"""
import numpy as np
from pokerStats.rl.poker_env import PokerEnv, Action, NUM_ACTIONS, OBS_DIM


def test_obs_shape():
    """Observation vectors have the correct dimension."""
    env = PokerEnv(num_players=6)
    obs = env.reset()
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
