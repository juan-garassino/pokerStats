"""Tests for pokerStats.rl.ppo_agent"""
import numpy as np
import torch
from pokerStats.rl.ppo_agent import PPOAgent, RolloutBuffer, AlphaPokerNet, NUM_CFR_ACTIONS
from pokerStats.rl.poker_env import OBS_DIM, NUM_ACTIONS, Action


def test_get_action_valid():
    """Agent returns a valid action within legal mask."""
    agent = PPOAgent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[Action.FOLD] = True
    mask[Action.CALL] = True

    action, raise_frac, log_prob, value, entropy = agent.get_action(obs, mask)
    assert action in [Action.FOLD, Action.CALL], f"Action {action} not in legal set"


def test_get_action_returns_raise_frac():
    """Agent returns raise_frac in [0, 1]."""
    agent = PPOAgent()
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    action, raise_frac, _, _, _ = agent.get_action(obs, mask)
    assert 0.0 <= raise_frac <= 1.0, f"raise_frac {raise_frac} out of range"


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
        buf.add(obs, 0, 0.5, 0.0, 0.0, 0.0, 0.0, mask)
    assert not buf.full

    buf.add(obs, 0, 0.5, 0.0, 0.0, 0.0, 0.0, mask)
    assert buf.full


def test_buffer_stores_raise_frac():
    """Buffer stores raise_frac values."""
    buf = RolloutBuffer(capacity=4)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    buf.add(obs, Action.RAISE, 0.73, 0.0, 0.0, 0.0, 0.0, mask)
    assert abs(buf.raise_fracs[0] - 0.73) < 1e-5


def test_buffer_reset():
    """Buffer resets correctly."""
    buf = RolloutBuffer(capacity=4)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    for _ in range(4):
        buf.add(obs, 0, 0.5, 0.0, 0.0, 0.0, 0.0, mask)
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
        buf.add(obs, Action.RAISE, 0.5, -0.5, 1.0, 0.5, 0.0, mask)

    metrics = agent.update(buf, ppo_epochs=1, batch_size=16)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "entropy" in metrics


def test_num_actions_is_five():
    """Discrete action space is 5 (fold/check/call/raise/all-in)."""
    assert NUM_ACTIONS == 5


def test_alpha_agent_get_action():
    """AlphaPoker agent returns valid actions."""
    agent = PPOAgent(use_alpha=True, num_opponents=2)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    action, raise_frac, log_prob, value, entropy = agent.get_action(obs, mask)
    assert 0 <= action < NUM_ACTIONS
    assert 0.0 <= raise_frac <= 1.0


def test_alpha_update_with_aux():
    """AlphaPoker agent update computes auxiliary losses."""
    from pokerStats.rl.opponent_encoder import EVENT_DIM, MAX_SEQ_LEN
    agent = PPOAgent(use_alpha=True, num_opponents=2)
    buf = RolloutBuffer(capacity=32, num_opponents=2)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    opp_ev = np.random.randn(2, MAX_SEQ_LEN, EVENT_DIM).astype(np.float32)
    opp_mk = np.zeros((2, MAX_SEQ_LEN), dtype=bool)

    for i in range(32):
        buf.add(obs, Action.RAISE, 0.5, -0.5, 1.0, 0.5, 0.0, mask, opp_ev, opp_mk)
        buf.has_showdown[i] = True
        buf.showdown_cards[i, 0, 0] = 1.0  # some card
        buf.next_actions[i, 0] = Action.CALL
        buf.has_next_action[i, 0] = True

    metrics = agent.update(buf, ppo_epochs=1, batch_size=16, hand_num=10000)
    assert "aux_card_loss" in metrics
    assert "aux_range_loss" in metrics
    assert "aux_action_loss" in metrics


def test_cfr_head_output_shape():
    """CFR distillation head produces (batch, NUM_CFR_ACTIONS) output."""
    net = AlphaPokerNet(num_opponents=2)
    obs = torch.randn(4, OBS_DIM)
    out = net(obs)
    # forward returns: logits, raise_alpha, raise_beta, value, latents, card_preds, range_pred, cfr_logits
    cfr_logits = out[7]
    assert cfr_logits.shape == (4, NUM_CFR_ACTIONS), \
        f"Expected (4, {NUM_CFR_ACTIONS}), got {cfr_logits.shape}"


def test_cfr_distillation_loss():
    """CFR distillation loss decreases when network matches blueprint targets."""
    from pokerStats.rl.opponent_encoder import EVENT_DIM, MAX_SEQ_LEN
    agent = PPOAgent(use_alpha=True, num_opponents=2)
    buf = RolloutBuffer(capacity=32, num_opponents=2)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    opp_ev = np.random.randn(2, MAX_SEQ_LEN, EVENT_DIM).astype(np.float32)
    opp_mk = np.zeros((2, MAX_SEQ_LEN), dtype=bool)

    # Uniform CFR target
    cfr_target = np.ones(NUM_CFR_ACTIONS, dtype=np.float32) / NUM_CFR_ACTIONS

    for i in range(32):
        buf.add(obs, Action.CALL, 0.5, -0.5, 1.0, 0.5, 0.0, mask,
                opp_ev, opp_mk, cfr_target=cfr_target, has_cfr=True)

    metrics = agent.update(buf, ppo_epochs=2, batch_size=16,
                           hand_num=10000, cfr_coef=1.0)
    assert "cfr_loss" in metrics
    assert metrics["cfr_loss"] >= 0.0


def test_cfr_coef_zero_no_effect():
    """With cfr_coef=0, CFR loss should be zero (same as pure PPO)."""
    from pokerStats.rl.opponent_encoder import EVENT_DIM, MAX_SEQ_LEN
    agent = PPOAgent(use_alpha=True, num_opponents=2)
    buf = RolloutBuffer(capacity=32, num_opponents=2)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    opp_ev = np.random.randn(2, MAX_SEQ_LEN, EVENT_DIM).astype(np.float32)
    opp_mk = np.zeros((2, MAX_SEQ_LEN), dtype=bool)
    cfr_target = np.ones(NUM_CFR_ACTIONS, dtype=np.float32) / NUM_CFR_ACTIONS

    for i in range(32):
        buf.add(obs, Action.CALL, 0.5, -0.5, 1.0, 0.5, 0.0, mask,
                opp_ev, opp_mk, cfr_target=cfr_target, has_cfr=True)

    metrics = agent.update(buf, ppo_epochs=1, batch_size=16,
                           hand_num=10000, cfr_coef=0.0)
    assert metrics["cfr_loss"] == 0.0, "cfr_coef=0 should produce zero CFR loss"


def test_buffer_stores_cfr_targets():
    """Buffer stores and yields CFR targets correctly."""
    buf = RolloutBuffer(capacity=4)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    cfr_target = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    buf.add(obs, 0, 0.5, 0.0, 0.0, 0.0, 0.0, mask,
            cfr_target=cfr_target, has_cfr=True)
    assert buf.has_cfr_target[0] is True or buf.has_cfr_target[0] == True
    np.testing.assert_allclose(buf.cfr_targets[0], cfr_target, atol=1e-6)

    # Without CFR target
    buf.add(obs, 0, 0.5, 0.0, 0.0, 0.0, 0.0, mask)
    assert not buf.has_cfr_target[1]
