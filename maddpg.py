"""
maddpg.py
─────────
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) – baseline version.

Key idea (Lowe et al., 2017):
  • Decentralised execution: each agent's Actor sees only ITS OWN observation.
  • Centralised training:   each agent's Critic sees ALL observations + ALL actions.

Networks
────────
  Actor  (obs_i)                       → action_i  ∈ [0,1]^act_dim   (Sigmoid output)
  Critic (obs_1‥N ⊕ act_1‥N)          → Q-value   ∈ ℝ

Training
────────
  Critic loss:  MSE( Q(o,a),  r_i + γ · Q_tgt(o', μ_tgt_1(o'_1), …) )
  Actor  loss:  -E[ Q_i(o, [μ_i(o_i), a_j≠i_from_replay]) ]

Soft target updates (Polyak averaging) keep training stable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── Networks ──────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Maps agent i's private observation → action in [0,1]^act_dim."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid(),   # actions bounded in [0, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Centralised critic: takes all observations + all actions → scalar Q."""

    def __init__(self, obs_dim_total: int, act_dim_total: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim_total + act_dim_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs_list: list[torch.Tensor],   # [n_agents] each [B, obs_dim_i]
        act_list: list[torch.Tensor],   # [n_agents] each [B, act_dim_i]
    ) -> torch.Tensor:
        x = torch.cat(obs_list + act_list, dim=-1)
        return self.net(x)                              # [B, 1]


# ── Agent ─────────────────────────────────────────────────────────────────────

class MADDPGAgent:
    """
    One MADDPG agent (one Actor + one Critic pair, each with a target network).

    Parameters
    ----------
    agent_idx     : index into the joint obs/act lists (0-based)
    obs_dim       : dimension of THIS agent's observation
    act_dim       : dimension of THIS agent's action
    obs_dim_total : sum of all agents' obs dims  (for Critic input)
    act_dim_total : sum of all agents' act dims  (for Critic input)
    """

    def __init__(
        self,
        agent_idx:     int,
        obs_dim:       int,
        act_dim:       int,
        obs_dim_total: int,
        act_dim_total: int,
        lr_actor:      float = 1e-3,
        lr_critic:     float = 1e-3,
        gamma:         float = 0.95,
        tau:           float = 0.01,
        hidden_dim:    int   = 128,
    ):
        self.agent_idx = agent_idx
        self.gamma     = gamma
        self.tau       = tau

        # ── Actor ────────────────────────────────────────────────────────────
        self.actor        = Actor(obs_dim, act_dim, hidden_dim)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # ── Critic ───────────────────────────────────────────────────────────
        self.critic        = Critic(obs_dim_total, act_dim_total, hidden_dim)
        self.target_critic = Critic(obs_dim_total, act_dim_total, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ── Action selection ─────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        Greedy action from the current Actor + optional Gaussian exploration noise.

        noise_scale = 0.0  →  fully deterministic (evaluation mode)
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)    # [1, obs_dim]
        action = self.actor(obs_t).squeeze(0).numpy()   # [act_dim]

        if noise_scale > 0.0:
            action = action + np.random.normal(0.0, noise_scale, action.shape)
            action = np.clip(action, 0.0, 1.0)

        return action.astype(np.float32)

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, batch: tuple, all_agents: list) -> tuple[float, float]:
        """
        Perform one gradient step for both Critic and Actor.

        Parameters
        ----------
        batch      : output of ReplayBuffer.sample()
                     (obs_n, act_n, rew_n, next_obs_n, done_n)
                     each element is a list of n_agents tensors [B, dim]
        all_agents : list of all MADDPGAgent instances (needed for actor update)

        Returns
        -------
        (critic_loss, actor_loss) as Python floats
        """
        obs_n, act_n, rew_n, next_obs_n, done_n = batch
        n = len(all_agents)
        i = self.agent_idx

        rew_i  = rew_n[i].unsqueeze(-1)    # [B, 1]
        done_i = done_n[i].unsqueeze(-1)   # [B, 1]

        # ── Critic update ────────────────────────────────────────────────────
        with torch.no_grad():
            target_acts = [
                all_agents[j].target_actor(next_obs_n[j]) for j in range(n)
            ]
            q_next   = self.target_critic(next_obs_n, target_acts)   # [B, 1]
            q_target = rew_i + self.gamma * q_next * (1.0 - done_i)  # Bellman

        q_pred      = self.critic(obs_n, act_n)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # ── Actor update ─────────────────────────────────────────────────────
        # Only agent i's actions are differentiated; all others are fixed from replay.
        curr_acts = []
        for j in range(n):
            if j == i:
                curr_acts.append(self.actor(obs_n[j]))          # differentiable
            else:
                curr_acts.append(act_n[j].detach())             # constant

        actor_loss = -self.critic(obs_n, curr_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # ── Soft target updates (Polyak averaging) ───────────────────────────
        self._soft_update(self.actor,  self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path_prefix: str):
        """Save actor and critic weights.  path_prefix e.g. 'checkpoints/buyer'"""
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pt"))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    # ── Private ───────────────────────────────────────────────────────────────

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        """θ_tgt ← τ·θ_src + (1−τ)·θ_tgt"""
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_tgt.data)
