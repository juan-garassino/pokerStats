"""
PPO agent with action masking + continuous raise sizing + opponent modeling
───────────────────────────────────────────────────────────────────────────
Architecture:
  - AlphaPokerNet: shared trunk + opponent encoder + auxiliary heads
    1. Action head:  5 discrete logits (fold/check/call/raise/all-in)
    2. Raise head:   Beta(alpha, beta) distribution for raise sizing [0,1]
    3. Critic head:  scalar value estimate
    4. Card predictor: 52-dim per opponent (auxiliary, training only)
    5. Action predictor: 5-dim per opponent (auxiliary, training only)
    6. Range head: num_opponents × 52 (auxiliary, training only)
  - Legacy PokerNet kept for backward compatibility
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from typing import Optional

from .poker_env import NUM_ACTIONS, OBS_DIM, MAX_PLAYERS, Action
from .opponent_encoder import OpponentEncoder, EVENT_DIM, MAX_SEQ_LEN, LATENT_DIM

# CFR action count — must match pokerStats.cfr.cfr_solver.NUM_CFR_ACTIONS
NUM_CFR_ACTIONS = 7


# ── Network architecture ──────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """Small residual block for the trunk."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class PokerNet(nn.Module):
    """
    Legacy actor-critic network (no opponent encoding).
    Kept for backward compatibility with old checkpoints.
    """
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.res3 = ResBlock(hidden)

        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.raise_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.raise_head[-1].bias, 1.0)

    def _trunk(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x

    def forward(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None):
        x = self._trunk(obs)

        logits = self.actor(x)
        value  = self.critic(x).squeeze(-1)

        raise_params = F.softplus(self.raise_head(x)) + 1.0
        raise_alpha  = raise_params[..., 0]
        raise_beta   = raise_params[..., 1]

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        return logits, raise_alpha, raise_beta, value

    def act(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> tuple:
        logits, raise_alpha, raise_beta, value = self.forward(obs, legal_mask)

        action_dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = action_dist.sample()
        log_prob_action = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        raise_dist = Beta(raise_alpha, raise_beta)
        if deterministic:
            raise_frac = raise_dist.mean
        else:
            raise_frac = raise_dist.sample()
        log_prob_raise = raise_dist.log_prob(raise_frac)

        is_raise = (action == Action.RAISE).float()
        log_prob = log_prob_action + is_raise * log_prob_raise

        return action, raise_frac, log_prob, value, entropy


class AlphaPokerNet(nn.Module):
    """
    Actor-Critic with opponent modeling, auxiliary heads, and range prediction.

    Input: obs (160) + opponent latents (8 × 32 = 256) → fused (416)
    Trunk: 3 ResBlocks (256 hidden)
    Heads: actor (5), raise (Beta α,β), critic (1)
    Aux:   card predictor (52/opp), action predictor (5/opp), range (8×52)
    Supports 2-9 players. Empty seats get zero latents.
    """
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256,
                 num_actions: int = NUM_ACTIONS, num_opponents: int = MAX_PLAYERS - 1,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.num_opponents = num_opponents
        self.latent_dim = latent_dim

        # Opponent encoder (shared across all seats)
        self.opponent_encoder = OpponentEncoder(latent_dim=latent_dim)

        # Combined input: obs + opponent latents
        combined_dim = obs_dim + num_opponents * latent_dim  # 145 + 160 = 305

        # Trunk
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.res3 = ResBlock(hidden)

        # Policy heads
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        self.raise_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Auxiliary heads (training only)
        self.card_predictor = nn.Linear(latent_dim, 52)
        self.action_predictor = nn.Sequential(
            nn.Linear(latent_dim + 64, num_actions),
        )
        # Game context projection for action predictor
        self.game_ctx_proj = nn.Linear(hidden, 64)
        self.range_head_proj = nn.Linear(hidden, num_opponents * 52)

        # CFR distillation head (predicts blueprint strategy from trunk)
        self.cfr_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, NUM_CFR_ACTIONS),
        )

        # Init
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.raise_head[-1].bias, 1.0)

    def _encode_opponents(self, opp_events: torch.Tensor,
                          opp_masks: torch.Tensor) -> list:
        """
        Encode each opponent's history into a latent vector.

        Args:
            opp_events: (batch, num_opponents, seq_len, EVENT_DIM)
            opp_masks:  (batch, num_opponents, seq_len)

        Returns: list of (batch, latent_dim) tensors, one per opponent
        """
        latents = []
        for i in range(self.num_opponents):
            lat = self.opponent_encoder(opp_events[:, i], opp_masks[:, i])
            latents.append(lat)
        return latents

    def _trunk(self, obs: torch.Tensor, opp_latent: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, opp_latent], dim=-1)
        x = self.input_proj(combined)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x

    def forward(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None,
                opp_events: Optional[torch.Tensor] = None,
                opp_masks: Optional[torch.Tensor] = None):
        """
        Full forward pass with opponent encoding.

        Returns: (logits, raise_alpha, raise_beta, value,
                  latents, card_preds, range_pred)
        """
        batch_size = obs.shape[0]

        # Encode opponents or use zeros
        if opp_events is not None and opp_masks is not None:
            latents = self._encode_opponents(opp_events, opp_masks)
            opp_latent = torch.cat(latents, dim=-1)
        else:
            latents = [torch.zeros(batch_size, self.latent_dim,
                                   device=obs.device) for _ in range(self.num_opponents)]
            opp_latent = torch.zeros(batch_size, self.num_opponents * self.latent_dim,
                                     device=obs.device)

        # Trunk
        x = self._trunk(obs, opp_latent)

        # Policy heads
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        raise_params = F.softplus(self.raise_head(x)) + 1.0
        raise_alpha = raise_params[..., 0]
        raise_beta = raise_params[..., 1]

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        # Auxiliary outputs
        card_preds = [torch.sigmoid(self.card_predictor(l)) for l in latents]

        game_ctx = F.relu(self.game_ctx_proj(x))
        range_pred = torch.sigmoid(self.range_head_proj(x))

        cfr_logits = self.cfr_head(x)

        return logits, raise_alpha, raise_beta, value, latents, card_preds, range_pred, cfr_logits

    def forward_with_trunk(self, obs, legal_mask=None, opp_events=None, opp_masks=None):
        """Forward pass that also returns trunk output for action prediction."""
        batch_size = obs.shape[0]

        if opp_events is not None and opp_masks is not None:
            latents = self._encode_opponents(opp_events, opp_masks)
            opp_latent = torch.cat(latents, dim=-1)
        else:
            latents = [torch.zeros(batch_size, self.latent_dim,
                                   device=obs.device) for _ in range(self.num_opponents)]
            opp_latent = torch.zeros(batch_size, self.num_opponents * self.latent_dim,
                                     device=obs.device)

        x = self._trunk(obs, opp_latent)

        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        raise_params = F.softplus(self.raise_head(x)) + 1.0
        raise_alpha = raise_params[..., 0]
        raise_beta = raise_params[..., 1]

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        card_preds = [torch.sigmoid(self.card_predictor(l)) for l in latents]
        game_ctx = F.relu(self.game_ctx_proj(x))
        range_pred = torch.sigmoid(self.range_head_proj(x))

        cfr_logits = self.cfr_head(x)

        return logits, raise_alpha, raise_beta, value, latents, card_preds, range_pred, cfr_logits, x

    def act(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None,
            opp_events: Optional[torch.Tensor] = None,
            opp_masks: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> tuple:
        """
        Sample action and raise sizing.
        Returns (action, raise_frac, log_prob, value, entropy).
        """
        logits, raise_alpha, raise_beta, value, latents, _, _, _ = self.forward(
            obs, legal_mask, opp_events, opp_masks
        )

        action_dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = action_dist.sample()
        log_prob_action = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        raise_dist = Beta(raise_alpha, raise_beta)
        if deterministic:
            raise_frac = raise_dist.mean
        else:
            raise_frac = raise_dist.sample()
        log_prob_raise = raise_dist.log_prob(raise_frac)

        is_raise = (action == Action.RAISE).float()
        log_prob = log_prob_action + is_raise * log_prob_raise

        return action, raise_frac, log_prob, value, entropy


# ── PPO rollout buffer ────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int = OBS_DIM,
                 num_actions: int = NUM_ACTIONS,
                 num_opponents: int = MAX_PLAYERS - 1, max_seq_len: int = MAX_SEQ_LEN,
                 event_dim: int = EVENT_DIM):
        self.capacity    = capacity
        self.obs         = np.zeros((capacity, obs_dim),      dtype=np.float32)
        self.actions     = np.zeros(capacity,                 dtype=np.int64)
        self.raise_fracs = np.zeros(capacity,                 dtype=np.float32)
        self.log_probs   = np.zeros(capacity,                 dtype=np.float32)
        self.rewards     = np.zeros(capacity,                 dtype=np.float32)
        self.values      = np.zeros(capacity,                 dtype=np.float32)
        self.dones       = np.zeros(capacity,                 dtype=np.float32)
        self.legal_masks = np.ones((capacity, num_actions),   dtype=bool)

        # Opponent context snapshot at each step
        self.opp_events  = np.zeros((capacity, num_opponents, max_seq_len, event_dim),
                                    dtype=np.float32)
        self.opp_masks   = np.ones((capacity, num_opponents, max_seq_len), dtype=bool)

        # Auxiliary targets (filled at end of hand)
        self.showdown_cards = np.zeros((capacity, num_opponents, 52), dtype=np.float32)
        self.has_showdown   = np.zeros(capacity, dtype=bool)
        self.next_actions   = np.zeros((capacity, num_opponents), dtype=np.int64)
        self.has_next_action = np.zeros((capacity, num_opponents), dtype=bool)

        # CFR distillation targets
        self.cfr_targets    = np.zeros((capacity, NUM_CFR_ACTIONS), dtype=np.float32)
        self.has_cfr_target = np.zeros(capacity, dtype=bool)

        self.ptr  = 0
        self.full = False

    def add(self, obs, action, raise_frac, log_prob, reward, value, done,
            legal_mask, opp_events=None, opp_masks=None,
            cfr_target=None, has_cfr=False):
        i = self.ptr % self.capacity
        self.obs[i]         = obs
        self.actions[i]     = action
        self.raise_fracs[i] = raise_frac
        self.log_probs[i]   = log_prob
        self.rewards[i]     = reward
        self.values[i]      = value
        self.dones[i]       = done
        self.legal_masks[i] = legal_mask
        if opp_events is not None:
            self.opp_events[i] = opp_events
        if opp_masks is not None:
            self.opp_masks[i] = opp_masks
        if cfr_target is not None:
            self.cfr_targets[i] = cfr_target
            self.has_cfr_target[i] = has_cfr
        else:
            self.has_cfr_target[i] = False
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def compute_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95,
                        last_value: float = 0.0) -> tuple:
        """GAE advantage estimation."""
        n         = self.capacity
        returns   = np.zeros(n, dtype=np.float32)
        advantages= np.zeros(n, dtype=np.float32)
        last_gae  = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n-1 else self.values[t+1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        advantages = np.clip(advantages, -10.0, 10.0)
        return returns, advantages

    def get_batches(self, batch_size: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        returns, advantages = self.compute_returns(gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.random.permutation(self.capacity)
        for start in range(0, self.capacity, batch_size):
            b = idx[start:start+batch_size]
            yield (
                torch.FloatTensor(self.obs[b]),
                torch.LongTensor(self.actions[b]),
                torch.FloatTensor(self.raise_fracs[b]),
                torch.FloatTensor(self.log_probs[b]),
                torch.FloatTensor(returns[b]),
                torch.FloatTensor(advantages[b]),
                torch.BoolTensor(self.legal_masks[b]),
                torch.FloatTensor(self.opp_events[b]),
                torch.BoolTensor(self.opp_masks[b]),
                torch.FloatTensor(self.showdown_cards[b]),
                torch.BoolTensor(self.has_showdown[b]),
                torch.FloatTensor(self.values[b]),
                torch.LongTensor(self.next_actions[b]),
                torch.BoolTensor(self.has_next_action[b]),
                torch.FloatTensor(self.cfr_targets[b]),
                torch.BoolTensor(self.has_cfr_target[b]),
            )

    def reset(self):
        self.ptr  = 0
        self.full = False


# ── PPO trainer ───────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(
        self,
        obs_dim:      int   = OBS_DIM,
        num_actions:  int   = NUM_ACTIONS,
        hidden:       int   = 256,
        lr:           float = 3e-4,
        clip_eps:     float = 0.2,
        entropy_coef: float = 0.01,
        value_coef:   float = 0.5,
        max_grad_norm:float = 0.5,
        device:       str   = "auto",
        use_alpha:    bool  = False,
        num_opponents:int   = MAX_PLAYERS - 1,
        aux_card_coef:  float = 0.5,
        aux_action_coef:float = 0.3,
        aux_range_coef: float = 0.3,
    ):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "mps"  if device == "auto" and torch.backends.mps.is_available() else
            "cpu"
        )
        self.use_alpha = use_alpha
        self.num_opponents = num_opponents

        if use_alpha:
            self.net = AlphaPokerNet(obs_dim, hidden, num_actions,
                                     num_opponents).to(self.device)
            self._cpu_net = AlphaPokerNet(obs_dim, hidden, num_actions,
                                          num_opponents)
        else:
            self.net = PokerNet(obs_dim, hidden, num_actions).to(self.device)
            self._cpu_net = PokerNet(obs_dim, hidden, num_actions)

        self.optimizer    = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.entropy_floor = 1.0        # adaptive boost kicks in below this (max ent for 5 actions ≈ 1.61)
        self.entropy_boost = 8.0        # multiplier when below floor
        self.value_coef   = value_coef
        self.max_grad_norm= max_grad_norm
        self.train_steps  = 0

        # Auxiliary loss coefficients
        self.aux_card_coef   = aux_card_coef
        self.aux_action_coef = aux_action_coef
        self.aux_range_coef  = aux_range_coef

        self._cpu_net.load_state_dict(self.net.state_dict())
        self._cpu_net.eval()
        self._cpu_dirty = False

        # ELO rating (for league tracking)
        self.elo = 1500.0

        # Running return normalization
        self._ret_mean = 0.0
        self._ret_var = 1.0
        self._ret_count = 0

    def _sync_cpu_net(self):
        """Copy GPU weights to CPU inference net."""
        if self._cpu_dirty:
            self._cpu_net.load_state_dict(
                {k: v.cpu() for k, v in self.net.state_dict().items()}
            )
            self._cpu_dirty = False

    @torch.inference_mode()
    def get_action(self, obs: np.ndarray, legal_mask: np.ndarray,
                   deterministic: bool = False,
                   opp_events: Optional[np.ndarray] = None,
                   opp_masks: Optional[np.ndarray] = None,
                   temperature: float = 1.0) -> tuple:
        """
        Returns (action, raise_frac, log_prob, value, entropy).
        Accepts optional opponent context for AlphaPokerNet.
        """
        self._sync_cpu_net()
        obs_t  = torch.from_numpy(obs).unsqueeze(0)
        mask_t = torch.from_numpy(legal_mask.astype(np.bool_)).unsqueeze(0)

        if self.use_alpha and opp_events is not None:
            opp_ev_t = torch.from_numpy(opp_events).unsqueeze(0)
            opp_mk_t = torch.from_numpy(opp_masks).unsqueeze(0)
            logits, raise_alpha, raise_beta, value, _, _, _, _ = self._cpu_net(
                obs_t, mask_t, opp_ev_t, opp_mk_t
            )
        elif self.use_alpha:
            logits, raise_alpha, raise_beta, value, _, _, _, _ = self._cpu_net(
                obs_t, mask_t
            )
        else:
            logits, raise_alpha, raise_beta, value = self._cpu_net(obs_t, mask_t)

        # Sample action using numpy (avoids torch Categorical overhead)
        logits_np = logits[0].numpy()
        logits_np = logits_np / max(temperature, 0.01)
        logits_np = logits_np - logits_np.max()
        probs = np.exp(logits_np)
        probs = probs / probs.sum()

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        log_prob_action = float(np.log(probs[action] + 1e-8))

        # Sample raise fraction using numpy Beta
        alpha = float(raise_alpha[0].numpy())
        beta_v = float(raise_beta[0].numpy())
        if deterministic:
            raise_frac = alpha / (alpha + beta_v)
        else:
            raise_frac = float(np.random.beta(alpha, beta_v))

        from math import lgamma, log
        rf_c = np.clip(raise_frac, 1e-6, 1 - 1e-6)
        log_prob_raise = (
            (alpha - 1) * log(rf_c) + (beta_v - 1) * log(1 - rf_c)
            + lgamma(alpha + beta_v) - lgamma(alpha) - lgamma(beta_v)
        )
        is_raise = 1.0 if action == Action.RAISE else 0.0
        log_prob = log_prob_action + is_raise * log_prob_raise

        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        return (action, raise_frac, log_prob, float(value[0].numpy()), entropy)

    def _aux_warmup(self, hand_num: int, warmup_hands: int = 5000) -> float:
        """Ramp auxiliary loss coefficient from 0 to 1 over warmup_hands."""
        return min(1.0, hand_num / max(warmup_hands, 1))

    def _update_ret_stats(self, returns: np.ndarray):
        """Welford's online algorithm for running mean/var."""
        for r in returns:
            self._ret_count += 1
            delta = r - self._ret_mean
            self._ret_mean += delta / self._ret_count
            delta2 = r - self._ret_mean
            self._ret_var += (delta * delta2 - self._ret_var) / self._ret_count

    def update(self, buffer: RolloutBuffer, ppo_epochs: int = 4,
               batch_size: int = 256, hand_num: int = 0,
               cfr_coef: float = 0.0) -> dict:
        """Run PPO update with optional auxiliary + CFR distillation losses. Returns metrics."""
        total_loss = total_policy = total_value = total_entropy = 0.0
        total_aux_card = total_aux_range = total_aux_action = 0.0
        total_cfr_loss = 0.0
        n_updates = 0

        warmup = self._aux_warmup(hand_num) if self.use_alpha else 0.0

        for _ in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size):
                if self.use_alpha:
                    obs, actions, raise_fracs, old_log_probs, returns, \
                        advantages, masks, opp_ev, opp_mk, sd_cards, has_sd, \
                        old_values, next_actions, has_next_act, \
                        cfr_targets_b, has_cfr_b = batch
                    obs, actions, raise_fracs, old_log_probs, returns, \
                        advantages, masks, opp_ev, opp_mk, sd_cards, has_sd, \
                        old_values, next_actions, has_next_act, \
                        cfr_targets_b, has_cfr_b = (
                        t.to(self.device) for t in
                        (obs, actions, raise_fracs, old_log_probs, returns,
                         advantages, masks, opp_ev, opp_mk, sd_cards, has_sd,
                         old_values, next_actions, has_next_act,
                         cfr_targets_b, has_cfr_b)
                    )

                    logits, raise_alpha, raise_beta, values, latents, \
                        card_preds, range_pred, cfr_logits, x = self.net.forward_with_trunk(obs, masks, opp_ev, opp_mk)
                else:
                    obs, actions, raise_fracs, old_log_probs, returns, \
                        advantages, masks, *_ = batch
                    obs, actions, raise_fracs, old_log_probs, returns, \
                        advantages, masks = (
                        t.to(self.device) for t in
                        (obs, actions, raise_fracs, old_log_probs, returns,
                         advantages, masks)
                    )
                    logits, raise_alpha, raise_beta, values = self.net(obs, masks)

                # NaN guard — skip corrupted batches
                # Note: -inf is expected in masked positions (legal_mask), only check for NaN
                if torch.isnan(logits).any():
                    print("  [WARN] NaN in logits — skipping batch")
                    continue

                # Discrete action log prob
                action_dist = Categorical(logits=logits)
                log_probs_action = action_dist.log_prob(actions)
                entropy = action_dist.entropy().mean()

                # Continuous raise log prob
                raise_dist = Beta(raise_alpha, raise_beta)
                rf_clamped = raise_fracs.clamp(1e-6, 1.0 - 1e-6)
                log_probs_raise = raise_dist.log_prob(rf_clamped)
                is_raise = (actions == Action.RAISE).float()
                log_probs = log_probs_action + is_raise * log_probs_raise

                # PPO-clip policy loss
                ratio = (log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with running normalization + clipping
                self._update_ret_stats(returns.cpu().numpy())
                ret_std = max(self._ret_var ** 0.5, 1e-4)
                returns_norm = (returns - self._ret_mean) / ret_std

                if self.use_alpha:
                    values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
                    v_loss1 = (values - returns_norm) ** 2
                    v_loss2 = (values_clipped - returns_norm) ** 2
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = F.mse_loss(values, returns_norm)

                # Total loss — adaptive entropy: boost coefficient when entropy collapses
                ent_coef = self.entropy_coef
                if entropy.item() < self.entropy_floor:
                    ent_coef = self.entropy_coef * self.entropy_boost
                loss = policy_loss + self.value_coef * value_loss - ent_coef * entropy

                # Auxiliary losses (AlphaPokerNet only)
                aux_card_loss = torch.tensor(0.0, device=self.device)
                aux_range_loss = torch.tensor(0.0, device=self.device)
                aux_action_loss = torch.tensor(0.0, device=self.device)

                if self.use_alpha and warmup > 0:
                    # Card prediction loss (only on showdown hands)
                    if has_sd.any():
                        sd_mask = has_sd.float()
                        for opp_i in range(self.num_opponents):
                            target = sd_cards[:, opp_i]  # (batch, 52)
                            pred = card_preds[opp_i]     # (batch, 52)
                            card_bce = F.binary_cross_entropy(pred, target,
                                                              reduction='none')
                            card_bce = (card_bce.mean(dim=-1) * sd_mask).sum() / \
                                       sd_mask.sum().clamp(min=1)
                            aux_card_loss = aux_card_loss + card_bce
                        aux_card_loss = aux_card_loss / self.num_opponents

                    # Range head loss (on showdown hands)
                    if has_sd.any():
                        sd_mask = has_sd.float()
                        range_target = sd_cards.reshape(sd_cards.shape[0], -1)
                        range_bce = F.binary_cross_entropy(
                            range_pred, range_target, reduction='none'
                        )
                        range_bce = (range_bce.mean(dim=-1) * sd_mask).sum() / \
                                    sd_mask.sum().clamp(min=1)
                        aux_range_loss = range_bce

                    # Action prediction loss (dense training signal for opponent encoder)
                    aux_action_loss = torch.tensor(0.0, device=self.device)
                    for opp_i in range(self.num_opponents):
                        has_target = has_next_act[:, opp_i]
                        if has_target.any():
                            game_ctx = F.relu(self.net.game_ctx_proj(x))
                            pred_input = torch.cat([latents[opp_i], game_ctx], dim=-1)
                            pred_logits = self.net.action_predictor(pred_input)
                            target = next_actions[:, opp_i]
                            act_ce = F.cross_entropy(pred_logits, target, reduction='none')
                            act_ce = (act_ce * has_target.float()).sum() / has_target.sum().clamp(min=1)
                            aux_action_loss = aux_action_loss + act_ce
                    aux_action_loss = aux_action_loss / max(self.num_opponents, 1)

                    loss = loss + warmup * (
                        self.aux_card_coef * aux_card_loss +
                        self.aux_range_coef * aux_range_loss +
                        self.aux_action_coef * aux_action_loss
                    )

                # CFR distillation loss (AlphaPokerNet only)
                cfr_loss = torch.tensor(0.0, device=self.device)
                if self.use_alpha and cfr_coef > 0 and has_cfr_b.any():
                    cfr_pred = F.log_softmax(cfr_logits[has_cfr_b], dim=-1)
                    cfr_tgt = cfr_targets_b[has_cfr_b]
                    # Clamp targets to valid probability distribution
                    cfr_tgt = cfr_tgt.clamp(min=1e-8)
                    cfr_tgt = cfr_tgt / cfr_tgt.sum(dim=-1, keepdim=True)
                    cfr_loss = F.kl_div(cfr_pred, cfr_tgt, reduction='batchmean',
                                        log_target=False)
                    loss = loss + cfr_coef * cfr_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss    += loss.item()
                total_policy  += policy_loss.item()
                total_value   += value_loss.item()
                total_entropy += entropy.item()
                total_aux_card += aux_card_loss.item()
                total_aux_range += aux_range_loss.item()
                total_aux_action += aux_action_loss.item()
                total_cfr_loss += cfr_loss.item()
                n_updates     += 1

        self.train_steps += 1
        self._cpu_dirty = True
        buffer.reset()

        metrics = {
            "loss":         total_loss    / max(n_updates, 1),
            "policy_loss":  total_policy  / max(n_updates, 1),
            "value_loss":   total_value   / max(n_updates, 1),
            "entropy":      total_entropy / max(n_updates, 1),
        }
        if self.use_alpha:
            metrics["aux_card_loss"]  = total_aux_card  / max(n_updates, 1)
            metrics["aux_range_loss"] = total_aux_range / max(n_updates, 1)
            metrics["aux_action_loss"] = total_aux_action / max(n_updates, 1)
            metrics["cfr_loss"]       = total_cfr_loss  / max(n_updates, 1)

        return metrics

    def save(self, path: str):
        torch.save({
            "net_state":   self.net.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "elo":         self.elo,
            "use_alpha":   self.use_alpha,
        }, path)
        print(f"  Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.train_steps = ckpt.get("train_steps", 0)
        self.elo         = ckpt.get("elo", 1500.0)
        self._cpu_dirty  = True
        print(f"  Loaded checkpoint <- {path}  (elo={self.elo:.0f})")
