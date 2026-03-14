"""
PPO agent with action masking
──────────────────────────────
Architecture:
  - Actor-Critic network (shared trunk + separate heads)
  - Action masking: illegal actions get -inf logits
  - Input: 143-dim observation vector
  - Output: 7 action logits + value estimate
  - Trained with PPO-clip + entropy bonus
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional

from .poker_env import NUM_ACTIONS, OBS_DIM


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
    Actor-Critic network for poker.

    Trunk processes the 143-dim obs into a 256-dim hidden state.
    Actor head outputs 7 action logits (masked for illegal actions).
    Critic head outputs a scalar value estimate.
    """
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()

        # Trunk: obs -> 256-dim representation
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.res3 = ResBlock(hidden)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Initialize actor output near zero (uniform initial policy)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None):
        x = self.input_proj(obs)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        logits = self.actor(x)
        value  = self.critic(x).squeeze(-1)

        # Mask illegal actions: set their logit to -inf before softmax
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))

        return logits, value

    def act(self, obs: torch.Tensor, legal_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> tuple:
        """Sample an action and return (action, log_prob, value, entropy)."""
        logits, value = self.forward(obs, legal_mask)
        dist  = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, value, entropy


# ── PPO rollout buffer ────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS):
        self.capacity    = capacity
        self.obs         = np.zeros((capacity, obs_dim),      dtype=np.float32)
        self.actions     = np.zeros(capacity,                 dtype=np.int64)
        self.log_probs   = np.zeros(capacity,                 dtype=np.float32)
        self.rewards     = np.zeros(capacity,                 dtype=np.float32)
        self.values      = np.zeros(capacity,                 dtype=np.float32)
        self.dones       = np.zeros(capacity,                 dtype=np.float32)
        self.legal_masks = np.ones((capacity, num_actions),   dtype=bool)
        self.ptr         = 0
        self.full        = False

    def add(self, obs, action, log_prob, reward, value, done, legal_mask):
        i = self.ptr % self.capacity
        self.obs[i]         = obs
        self.actions[i]     = action
        self.log_probs[i]   = log_prob
        self.rewards[i]     = reward
        self.values[i]      = value
        self.dones[i]       = done
        self.legal_masks[i] = legal_mask
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
                torch.FloatTensor(self.log_probs[b]),
                torch.FloatTensor(returns[b]),
                torch.FloatTensor(advantages[b]),
                torch.BoolTensor(self.legal_masks[b]),
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
    ):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "mps"  if device == "auto" and torch.backends.mps.is_available() else
            "cpu"
        )
        self.net          = PokerNet(obs_dim, hidden, num_actions).to(self.device)
        self.optimizer    = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.max_grad_norm= max_grad_norm
        self.train_steps  = 0

        # ELO rating (for league tracking)
        self.elo = 1500.0

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, legal_mask: np.ndarray,
                   deterministic: bool = False) -> tuple:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_t = torch.BoolTensor(legal_mask).unsqueeze(0).to(self.device)
        action, log_prob, value, entropy = self.net.act(obs_t, mask_t, deterministic)
        return (
            action.item(),
            log_prob.item(),
            value.item(),
            entropy.item(),
        )

    def update(self, buffer: RolloutBuffer, ppo_epochs: int = 4, batch_size: int = 256) -> dict:
        """Run PPO update. Returns dict of training metrics."""
        total_loss = total_policy = total_value = total_entropy = 0.0
        n_updates  = 0

        for _ in range(ppo_epochs):
            for obs, actions, old_log_probs, returns, advantages, masks in \
                    buffer.get_batches(batch_size):
                obs, actions, old_log_probs, returns, advantages, masks = (
                    t.to(self.device) for t in
                    (obs, actions, old_log_probs, returns, advantages, masks)
                )

                logits, values = self.net(obs, masks)
                dist    = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy   = dist.entropy().mean()

                # Policy loss (PPO-clip)
                ratio      = (log_probs - old_log_probs).exp()
                surr1      = ratio * advantages
                surr2      = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss    += loss.item()
                total_policy  += policy_loss.item()
                total_value   += value_loss.item()
                total_entropy += entropy.item()
                n_updates     += 1

        self.train_steps += 1
        buffer.reset()

        return {
            "loss":         total_loss    / n_updates,
            "policy_loss":  total_policy  / n_updates,
            "value_loss":   total_value   / n_updates,
            "entropy":      total_entropy / n_updates,
        }

    def save(self, path: str):
        torch.save({
            "net_state":   self.net.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "elo":         self.elo,
        }, path)
        print(f"  Saved checkpoint -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.train_steps = ckpt.get("train_steps", 0)
        self.elo         = ckpt.get("elo", 1500.0)
        print(f"  Loaded checkpoint <- {path}  (elo={self.elo:.0f})")
