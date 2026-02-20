
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maddpg import Critic   




class OpponentModel(nn.Module):
    def __init__(
        self,
        opp_act_dim:  int,
        history_len:  int,
        latent_dim:   int,
        hidden_dim:   int = 64,
    ):
        super().__init__()
        input_dim = opp_act_dim * history_len
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, opp_act_dim),
            nn.Sigmoid(),
        )

    def forward(self, history: torch.Tensor):
        psi  = self.encoder(history)
        pred = self.pred_head(psi)
        return psi, pred


class ToMActor(nn.Module):
   

    def __init__(
        self,
        obs_dim:    int,
        latent_dim: int,
        act_dim:    int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, psi], dim=-1))



class ToMMADDPGAgent:
  

    def __init__(
        self,
        agent_idx:       int,
        obs_dim:         int,
        act_dim:         int,
        obs_dim_total:   int,
        act_dim_total:   int,
        opp_act_dim:     int,
        opp_history_len: int   = 5,
        latent_dim:      int   = 16,
        lr_actor:        float = 1e-3,
        lr_critic:       float = 1e-3,
        gamma:           float = 0.95,
        tau:             float = 0.01,
        hidden_dim:      int   = 128,
        tom_loss_weight: float = 0.5,
    ):
        self.agent_idx       = agent_idx
        self.gamma           = gamma
        self.tau             = tau
        self.opp_history_len = opp_history_len
        self.opp_act_dim     = opp_act_dim
        self.latent_dim      = latent_dim
        self.tom_loss_weight = tom_loss_weight

        
        self.opp_model        = OpponentModel(opp_act_dim, opp_history_len, latent_dim)
        self.target_opp_model = OpponentModel(opp_act_dim, opp_history_len, latent_dim)
        self.target_opp_model.load_state_dict(self.opp_model.state_dict())

    
        self.actor        = ToMActor(obs_dim, latent_dim, act_dim, hidden_dim)
        self.target_actor = ToMActor(obs_dim, latent_dim, act_dim, hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())

  
        self.actor_opt = optim.Adam(
            list(self.actor.parameters()) + list(self.opp_model.parameters()),
            lr=lr_actor,
        )

  
        self.critic        = Critic(obs_dim_total, act_dim_total, hidden_dim)
        self.target_critic = Critic(obs_dim_total, act_dim_total, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt    = optim.Adam(self.critic.parameters(), lr=lr_critic)

   

    @torch.no_grad()
    def select_action(
        self,
        obs:         np.ndarray,
        opp_history: np.ndarray,   
        noise_scale: float = 0.0,
    ) -> np.ndarray:
      
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)
        hist_t = torch.FloatTensor(opp_history).unsqueeze(0)
        psi, _ = self.opp_model(hist_t)
        action = self.actor(obs_t, psi).squeeze(0).numpy()

        if noise_scale > 0.0:
            action = action + np.random.normal(0.0, noise_scale, action.shape)
            action = np.clip(action, 0.0, 1.0)

        return action.astype(np.float32)

   

    def update(
        self,
        batch:      tuple,
        all_agents: list,
    ) -> tuple[float, float, float]:
       
        obs_n, act_n, rew_n, next_obs_n, done_n, opp_hist_n = batch
        n = len(all_agents)
        i = self.agent_idx
        j = 1 - i  
        B = obs_n[0].shape[0]

        rew_i  = rew_n[i].unsqueeze(-1)
        done_i = done_n[i].unsqueeze(-1)

        
        with torch.no_grad():
            target_acts = []
            for k in range(n):
                k_opp = 1 - k
                opp_hist_3d_k = opp_hist_n[k].view(
                    B, self.opp_history_len, self.opp_act_dim
                )
                next_opp_hist_k = torch.cat(
                    [opp_hist_3d_k[:, 1:, :], act_n[k_opp].unsqueeze(1)], dim=1
                ).view(B, self.opp_history_len * self.opp_act_dim)

                tgt_psi_k, _ = all_agents[k].target_opp_model(next_opp_hist_k)
                target_acts.append(
                    all_agents[k].target_actor(next_obs_n[k], tgt_psi_k)
                )

            q_next   = self.target_critic(next_obs_n, target_acts)
            q_target = rew_i + self.gamma * q_next * (1.0 - done_i)

        q_pred      = self.critic(obs_n, act_n)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        
        psi_i, pred_opp_act = self.opp_model(opp_hist_n[i])

        
        opp_model_loss = F.mse_loss(pred_opp_act, act_n[j].detach())

        curr_acts = []
        for k in range(n):
            if k == i:
                curr_acts.append(self.actor(obs_n[k], psi_i))   # differentiable
            else:
                curr_acts.append(act_n[k].detach())

        actor_loss = -self.critic(obs_n, curr_acts).mean()


        joint_loss = actor_loss + self.tom_loss_weight * opp_model_loss

        self.actor_opt.zero_grad()
        joint_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.opp_model.parameters()),
            max_norm=1.0,
        )
        self.actor_opt.step()

        
        self._soft_update(self.actor,     self.target_actor)
        self._soft_update(self.critic,    self.target_critic)
        self._soft_update(self.opp_model, self.target_opp_model)

        return critic_loss.item(), actor_loss.item(), opp_model_loss.item()


    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(),     f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(),    f"{path_prefix}_critic.pt")
        torch.save(self.opp_model.state_dict(), f"{path_prefix}_opp_model.pt")

    def load(self, path_prefix: str):
        self.actor.load_state_dict(    torch.load(f"{path_prefix}_actor.pt"))
        self.critic.load_state_dict(   torch.load(f"{path_prefix}_critic.pt"))
        self.opp_model.load_state_dict(torch.load(f"{path_prefix}_opp_model.pt"))
        self.target_actor.load_state_dict(    self.actor.state_dict())
        self.target_critic.load_state_dict(   self.critic.state_dict())
        self.target_opp_model.load_state_dict(self.opp_model.state_dict())


    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_tgt.data)

