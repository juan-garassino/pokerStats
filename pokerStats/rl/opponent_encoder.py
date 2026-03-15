"""
Opponent encoder — transformer-based opponent modeling
───────────────────────────────────────────────────────
Components:
  - HandEvent vector (EVENT_DIM=65): encodes what one opponent did in one hand
  - InHandRecorder: tracks actions during a hand for all players
  - OpponentEncoder: transformer that maps event sequences → latent vectors
"""

import numpy as np
import torch
import torch.nn as nn
import math

from .poker_env import NUM_ACTIONS, Action

EVENT_DIM = 65
MAX_SEQ_LEN = 32
LATENT_DIM = 32


class InHandRecorder:
    """
    Accumulates per-player action data during a hand.
    After the hand ends, call build_event() to produce a HandEvent vector.
    """

    def __init__(self, num_players: int = 6):
        self.num_players = num_players
        self.reset()

    def reset(self):
        self.action_counts = np.zeros((self.num_players, NUM_ACTIONS), dtype=np.float32)
        self.raise_fracs = [[] for _ in range(self.num_players)]
        self.max_bet_frac = np.zeros(self.num_players, dtype=np.float32)
        self.street_reached = np.zeros(self.num_players, dtype=np.int32)
        self.total_actions = np.zeros(self.num_players, dtype=np.int32)

    def record(self, player_idx: int, action: int, raise_frac: float,
               pot: float, street_idx: int):
        """Record one action taken by player_idx."""
        self.action_counts[player_idx, action] += 1
        self.total_actions[player_idx] += 1
        self.street_reached[player_idx] = max(
            self.street_reached[player_idx], street_idx
        )
        if action == Action.RAISE and pot > 0:
            self.raise_fracs[player_idx].append(raise_frac)
            bet_frac = raise_frac  # already 0-1
            self.max_bet_frac[player_idx] = max(
                self.max_bet_frac[player_idx], bet_frac
            )
        elif action == Action.ALL_IN and pot > 0:
            self.max_bet_frac[player_idx] = 1.0

    def build_event(self, observer_idx: int, player_idx: int, env) -> np.ndarray:
        """
        After hand ends, encode what observer saw player do.

        Layout (EVENT_DIM=65):
          [0:5]   action counts (fold/check/call/raise/allin), normalized
          [5]     avg raise fraction when they raised
          [6]     max bet as fraction of pot
          [7]     position (0-1 normalized)
          [8]     street reached (0=preflop, 0.33=flop, 0.66=turn, 1.0=river)
          [9]     went to showdown (0 or 1)
          [10]    won the hand (0 or 1)
          [11]    pot contribution / starting stack
          [12]    hand was bluff (1 if showdown + weak hand + aggressive line)
          [13:65] cards revealed at showdown (52-dim one-hot, zeros if mucked)
        """
        v = np.zeros(EVENT_DIM, dtype=np.float32)
        p = env.players[player_idx]

        # [0:5] normalized action counts
        total = max(self.total_actions[player_idx], 1)
        v[0:5] = self.action_counts[player_idx] / total

        # [5] avg raise fraction
        rfs = self.raise_fracs[player_idx]
        v[5] = np.mean(rfs) if rfs else 0.0

        # [6] max bet fraction
        v[6] = self.max_bet_frac[player_idx]

        # [7] position normalized
        pos = (player_idx - env.dealer_idx) % env.num_players
        v[7] = pos / max(env.num_players - 1, 1)

        # [8] street reached
        v[8] = self.street_reached[player_idx] / 3.0

        # [9] went to showdown
        showdown = not p.folded and env._done
        v[9] = float(showdown)

        # [10] won the hand
        reward = env._rewards.get(player_idx, 0.0)
        v[10] = float(reward > 0)

        # [11] pot contribution / starting stack
        v[11] = p.total_bet / max(env.starting_stack, 1.0)

        # [12] bluff detection heuristic
        # aggressive line (raised) + went to showdown + weak cards
        was_aggressive = self.action_counts[player_idx, Action.RAISE] > 0 or \
                         self.action_counts[player_idx, Action.ALL_IN] > 0
        if showdown and was_aggressive:
            from .poker_env import hand_strength
            hs = hand_strength(p.hole_cards, env.community)
            v[12] = float(hs < 0.3)  # weak hand = likely bluff

        # [13:65] cards revealed at showdown (52 one-hot)
        if showdown and p.hole_cards:
            from .poker_env import cards_to_onehot
            v[13:65] = cards_to_onehot(p.hole_cards)

        return v


class OpponentEncoder(nn.Module):
    """
    Processes sequence of HandEvents for ONE opponent → latent vector.
    Shared across all seats. Called once per opponent per decision.

    Architecture: small transformer (2 layers, 4 heads, d_model=64).
    ~50K params.
    """

    def __init__(
        self,
        event_dim: int = EVENT_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_len: int = MAX_SEQ_LEN,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Project event vectors to d_model
        self.event_proj = nn.Linear(event_dim, d_model)

        # Learnable positional embeddings (index 0 = most recent)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Project to latent dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, events: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events: (batch, seq_len, EVENT_DIM) — hand event vectors
            padding_mask: (batch, seq_len) — True where padded (no event)

        Returns: (batch, LATENT_DIM) — opponent latent vector
        """
        batch_size, seq_len, _ = events.shape
        latent_dim = self.output_proj[0].out_features

        # Which rows have at least one real event?
        has_valid = (~padding_mask).any(dim=1)  # (batch,)

        # If no rows have history, return zeros (no transformer needed)
        if not has_valid.any():
            return torch.zeros(batch_size, latent_dim, device=events.device)

        # Only run transformer on rows with real history — avoids NaN from
        # fully-padded rows during backprop through attention weights
        valid_idx = has_valid.nonzero(as_tuple=True)[0]
        valid_events = events[valid_idx]       # (n_valid, seq, EVENT_DIM)
        valid_mask = padding_mask[valid_idx]    # (n_valid, seq)

        # Project events
        x = self.event_proj(valid_events)  # (n_valid, seq, d_model)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=events.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Transformer with padding mask (no fully-padded rows in this subset)
        x = self.transformer(x, src_key_padding_mask=valid_mask)

        # Pool: mean over non-padded positions
        mask_expanded = (~valid_mask).unsqueeze(-1).float()  # (n_valid, seq, 1)
        n_valid_tokens = mask_expanded.sum(dim=1).clamp(min=1)  # (n_valid, 1)
        pooled = (x * mask_expanded).sum(dim=1) / n_valid_tokens  # (n_valid, d_model)

        valid_latents = self.output_proj(pooled)  # (n_valid, latent_dim)

        # Stitch back: zeros for empty-history rows, encoder output for valid rows
        output = torch.zeros(batch_size, latent_dim, device=events.device)
        output[valid_idx] = valid_latents

        return output
