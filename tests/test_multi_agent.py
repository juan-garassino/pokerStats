"""Tests for multi-agent training loop"""
import numpy as np
from pokerStats.rl.poker_env import PokerEnv, Action, NUM_ACTIONS, OBS_DIM
from pokerStats.rl.ppo_agent import PPOAgent, RolloutBuffer, AlphaPokerNet
from pokerStats.rl.opponent_encoder import InHandRecorder, EVENT_DIM, MAX_SEQ_LEN
from pokerStats.rl.self_play_trainer import MultiAgentTrainer, CFG


def test_multi_agent_runs_hands():
    """MultiAgentTrainer can run 10 hands without crashing."""
    cfg = dict(CFG)
    cfg["phase1_hands"] = 10
    cfg["phase2_hands"] = 0
    cfg["phase3_hands"] = 0
    cfg["rollout_steps"] = 128
    cfg["batch_size"] = 32
    cfg["log_freq"] = 100
    cfg["eval_freq"] = 100
    cfg["save_freq"] = 100
    cfg["render_every"] = 100
    cfg["num_players"] = 3

    trainer = MultiAgentTrainer(cfg)
    trainer.train()
    assert trainer.hand_num >= 10


def test_buffers_fill():
    """Per-seat buffers accumulate steps during multi-agent play."""
    cfg = dict(CFG)
    cfg["phase1_hands"] = 20
    cfg["phase2_hands"] = 0
    cfg["phase3_hands"] = 0
    cfg["rollout_steps"] = 2048
    cfg["log_freq"] = 100
    cfg["eval_freq"] = 100
    cfg["save_freq"] = 100
    cfg["render_every"] = 100
    cfg["num_players"] = 3

    trainer = MultiAgentTrainer(cfg)
    trainer.train()

    # At least one buffer should have accumulated some steps
    total_steps = sum(buf.ptr for buf in trainer.buffers)
    assert total_steps > 0


def test_alpha_poker_net_forward():
    """AlphaPokerNet forward pass produces correct output shapes."""
    import torch
    net = AlphaPokerNet(num_opponents=5)
    batch = 4
    obs = torch.randn(batch, OBS_DIM)
    mask = torch.ones(batch, NUM_ACTIONS, dtype=torch.bool)
    opp_events = torch.randn(batch, 5, MAX_SEQ_LEN, EVENT_DIM)
    opp_masks = torch.zeros(batch, 5, MAX_SEQ_LEN, dtype=torch.bool)

    logits, ra, rb, value, latents, card_preds, range_pred = net(
        obs, mask, opp_events, opp_masks
    )

    assert logits.shape == (batch, NUM_ACTIONS)
    assert ra.shape == (batch,)
    assert rb.shape == (batch,)
    assert value.shape == (batch,)
    assert len(latents) == 5
    assert latents[0].shape == (batch, 32)
    assert len(card_preds) == 5
    assert card_preds[0].shape == (batch, 52)
    assert range_pred.shape == (batch, 5 * 52)


def test_alpha_poker_net_no_opp():
    """AlphaPokerNet works without opponent context (zeros fallback)."""
    import torch
    net = AlphaPokerNet(num_opponents=5)
    batch = 2
    obs = torch.randn(batch, OBS_DIM)
    mask = torch.ones(batch, NUM_ACTIONS, dtype=torch.bool)

    logits, ra, rb, value, latents, card_preds, range_pred = net(obs, mask)
    assert logits.shape == (batch, NUM_ACTIONS)


def test_opp_context_shape():
    """_get_opp_context returns correct shapes."""
    cfg = dict(CFG)
    cfg["num_players"] = 3
    cfg["phase1_hands"] = 0
    cfg["phase2_hands"] = 0
    cfg["phase3_hands"] = 0
    cfg["rollout_steps"] = 128

    trainer = MultiAgentTrainer(cfg)
    opp_events, opp_masks = trainer._get_opp_context(observer_idx=0)

    assert opp_events.shape == (2, MAX_SEQ_LEN, EVENT_DIM)  # 2 opponents
    assert opp_masks.shape == (2, MAX_SEQ_LEN)
    # All should be padded initially (no history yet)
    assert opp_masks.all()


def test_showdown_info():
    """Env step returns showdown info when hand ends at showdown."""
    env = PokerEnv(num_players=3)
    obs = env.reset()
    done = False
    while not done:
        legal = env.legal_actions()
        # Use CALL to get to showdown
        if Action.CALL in legal:
            action = Action.CALL
        elif Action.CHECK in legal:
            action = Action.CHECK
        else:
            action = legal[0]
        obs, rewards, done, info = env.step(action)

    # Hand completed
    assert done
    # Info should have showdown fields
    assert "showdown" in info
    assert "revealed_cards" in info
    assert "action_player" in info


def test_env_clone():
    """env.clone() creates independent copy."""
    env = PokerEnv(num_players=3)
    env.reset()
    # Make some moves
    legal = env.legal_actions()
    env.step(legal[0])

    clone = env.clone()

    # Modify clone
    clone.pot = 9999.0
    assert env.pot != 9999.0

    # Clone players are independent
    clone.players[0].stack = 0.0
    assert env.players[0].stack != 0.0


def test_rollout_buffer_with_opp_context():
    """RolloutBuffer stores opponent context arrays."""
    num_opp = 2
    buf = RolloutBuffer(capacity=4, num_opponents=num_opp)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    opp_ev = np.random.randn(num_opp, MAX_SEQ_LEN, EVENT_DIM).astype(np.float32)
    opp_mk = np.zeros((num_opp, MAX_SEQ_LEN), dtype=bool)

    buf.add(obs, 0, 0.5, 0.0, 0.0, 0.0, 0.0, mask, opp_ev, opp_mk)

    assert np.allclose(buf.opp_events[0], opp_ev)
    assert np.array_equal(buf.opp_masks[0], opp_mk)
