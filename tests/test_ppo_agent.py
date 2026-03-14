"""Tests for pokerStats.rl.ppo_agent"""
import numpy as np
from pokerStats.rl.ppo_agent import PPOAgent, RolloutBuffer
from pokerStats.rl.poker_env import OBS_DIM, NUM_ACTIONS


def test_get_action_valid():
    """Agent returns a valid action within legal mask."""
    agent = PPOAgent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[0] = True  # fold
    mask[2] = True  # call

    action, log_prob, value, entropy = agent.get_action(obs, mask)
    assert action in [0, 2], f"Action {action} not in legal set"


def test_get_action_deterministic():
    """Deterministic mode returns consistent actions."""
    agent = PPOAgent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    actions = [agent.get_action(obs, mask, deterministic=True)[0] for _ in range(10)]
    assert len(set(actions)) == 1, "Deterministic mode should return same action"


def test_buffer_add_and_full():
    """Buffer tracks capacity correctly."""
    buf = RolloutBuffer(capacity=4)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    for i in range(3):
        buf.add(obs, 0, 0.0, 0.0, 0.0, 0.0, mask)
    assert not buf.full

    buf.add(obs, 0, 0.0, 0.0, 0.0, 0.0, mask)
    assert buf.full


def test_buffer_reset():
    """Buffer resets correctly."""
    buf = RolloutBuffer(capacity=4)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    for _ in range(4):
        buf.add(obs, 0, 0.0, 0.0, 0.0, 0.0, mask)
    assert buf.full

    buf.reset()
    assert not buf.full
    assert buf.ptr == 0


def test_update_returns_metrics():
    """PPO update returns a dict of training metrics."""
    agent = PPOAgent()
    buf = RolloutBuffer(capacity=32)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    for _ in range(32):
        buf.add(obs, 0, -0.5, 1.0, 0.5, 0.0, mask)

    metrics = agent.update(buf, ppo_epochs=1, batch_size=16)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "entropy" in metrics
