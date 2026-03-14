"""Tests for pokerStats.rl.opponent_encoder"""
import numpy as np
import torch
from pokerStats.rl.opponent_encoder import (
    OpponentEncoder, InHandRecorder, EVENT_DIM, MAX_SEQ_LEN, LATENT_DIM,
)
from pokerStats.rl.poker_env import PokerEnv, Action


def test_event_dim():
    """EVENT_DIM is 65."""
    assert EVENT_DIM == 65


def test_recorder_reset():
    """InHandRecorder resets cleanly."""
    rec = InHandRecorder(num_players=6)
    rec.record(0, Action.RAISE, 0.5, 10.0, 0)
    rec.reset()
    assert rec.total_actions[0] == 0


def test_recorder_record():
    """InHandRecorder tracks action counts."""
    rec = InHandRecorder(num_players=3)
    rec.record(0, Action.CALL, 0.0, 10.0, 0)
    rec.record(0, Action.RAISE, 0.5, 20.0, 1)
    assert rec.total_actions[0] == 2
    assert rec.action_counts[0, Action.CALL] == 1
    assert rec.action_counts[0, Action.RAISE] == 1


def test_build_event_shape():
    """build_event produces a vector of shape (EVENT_DIM,)."""
    env = PokerEnv(num_players=3)
    env.reset()
    # Play to completion
    done = False
    rec = InHandRecorder(num_players=3)
    while not done:
        legal = env.legal_actions()
        action = legal[0]
        rec.record(env.current_player, action, 0.3, env.pot, env.street_idx)
        _, _, done, _ = env.step(action)

    event = rec.build_event(observer_idx=0, player_idx=1, env=env)
    assert event.shape == (EVENT_DIM,)
    assert event.dtype == np.float32


def test_build_event_values():
    """build_event populates expected fields."""
    env = PokerEnv(num_players=3)
    env.reset()
    rec = InHandRecorder(num_players=3)
    done = False
    while not done:
        legal = env.legal_actions()
        action = legal[0]
        rec.record(env.current_player, action, 0.3, env.pot, env.street_idx)
        _, _, done, _ = env.step(action)

    event = rec.build_event(0, 1, env)
    # Action counts should sum to ~1 (normalized)
    assert abs(event[0:5].sum() - 1.0) < 0.01 or event[0:5].sum() == 0.0
    # Position should be in [0, 1]
    assert 0.0 <= event[7] <= 1.0
    # Street reached in [0, 1]
    assert 0.0 <= event[8] <= 1.0


def test_encoder_output_shape():
    """OpponentEncoder produces (batch, LATENT_DIM) output."""
    encoder = OpponentEncoder()
    batch_size = 4
    seq_len = 8
    events = torch.randn(batch_size, seq_len, EVENT_DIM)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    output = encoder(events, padding_mask)
    assert output.shape == (batch_size, LATENT_DIM)


def test_encoder_with_padding():
    """OpponentEncoder handles padded sequences correctly."""
    encoder = OpponentEncoder()
    batch_size = 2
    seq_len = MAX_SEQ_LEN
    events = torch.randn(batch_size, seq_len, EVENT_DIM)
    # Pad last 20 positions
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, 12:] = True

    output = encoder(events, padding_mask)
    assert output.shape == (batch_size, LATENT_DIM)
    assert not torch.isnan(output).any()


def test_encoder_fully_padded():
    """OpponentEncoder handles fully padded (empty history) gracefully."""
    encoder = OpponentEncoder()
    events = torch.zeros(1, MAX_SEQ_LEN, EVENT_DIM)
    padding_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.bool)  # all padded

    output = encoder(events, padding_mask)
    assert output.shape == (1, LATENT_DIM)
    assert not torch.isnan(output).any()


def test_encoder_deterministic():
    """Same input produces same output."""
    encoder = OpponentEncoder()
    encoder.eval()
    events = torch.randn(2, 5, EVENT_DIM)
    mask = torch.zeros(2, 5, dtype=torch.bool)

    with torch.no_grad():
        out1 = encoder(events, mask)
        out2 = encoder(events, mask)
    assert torch.allclose(out1, out2)
