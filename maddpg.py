import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
 
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid(),  
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):

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
        obs_list: list[torch.Tensor],   
        act_list: list[torch.Tensor],   
    ) -> torch.Tensor:
        x = torch.cat(obs_list + act_list, dim=-1)
        return self.net(x)                              




class MADDPGAgent:

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

        
        self.actor        = Actor(obs_dim, act_dim, hidden_dim)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)

 
        self.critic        = Critic(obs_dim_total, act_dim_total, hidden_dim)
        self.target_critic = Critic(obs_dim_total, act_dim_total, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

 

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)    
        action = self.actor(obs_t).squeeze(0).numpy()   

        if noise_scale > 0.0:
            action = action + np.random.normal(0.0, noise_scale, action.shape)
            action = np.clip(action, 0.0, 1.0)

        return action.astype(np.float32)



    def update(self, batch: tuple, all_agents: list) -> tuple[float, float]:
        obs_n, act_n, rew_n, next_obs_n, done_n = batch
        n = len(all_agents)
        i = self.agent_idx
    
        rew_i  = rew_n[i].unsqueeze(-1)   
        done_i = done_n[i].unsqueeze(-1)   

    
        with torch.no_grad():
            target_acts = [
                all_agents[j].target_actor(next_obs_n[j]) for j in range(n)
            ]
            q_next   = self.target_critic(next_obs_n, target_acts)   
            q_target = rew_i + self.gamma * q_next * (1.0 - done_i)  

        q_pred      = self.critic(obs_n, act_n)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        curr_acts = []
        for j in range(n):
            if j == i:
                curr_acts.append(self.actor(obs_n[j]))         
            else:
                curr_acts.append(act_n[j].detach())             

        actor_loss = -self.critic(obs_n, curr_acts).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        self._soft_update(self.actor,  self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()


    def save(self, path_prefix: str):
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pt"))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_tgt.data)
