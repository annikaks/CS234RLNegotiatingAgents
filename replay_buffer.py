import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def add(
        self,
        obs_n:      list,   
        act_n:      list,  
        rew_n:      list,  
        next_obs_n: list,   
        done_n:     list,  
    ):
        self.buffer.append((
            [np.array(o, dtype=np.float32) for o in obs_n],
            [np.array(a, dtype=np.float32) for a in act_n],
            [float(r)                      for r in rew_n],
            [np.array(o, dtype=np.float32) for o in next_obs_n],
            [float(d)                      for d in done_n],
        ))

    def sample(self, batch_size: int):
        batch    = random.sample(self.buffer, batch_size)
        obs_n, act_n, rew_n, next_obs_n, done_n = zip(*batch)

        n_agents = len(obs_n[0])

        def _to_tensor_list(data, squeeze=False):
            out = []
            for i in range(n_agents):
                arr = np.stack([data[b][i] for b in range(batch_size)])
                t   = torch.FloatTensor(arr)
                if squeeze:
                    t = t.squeeze(-1)
                out.append(t)
            return out

        obs_tensors      = _to_tensor_list(obs_n)
        act_tensors      = _to_tensor_list(act_n)
        next_obs_tensors = _to_tensor_list(next_obs_n)

        rew_tensors  = [
            torch.FloatTensor([rew_n[b][i]  for b in range(batch_size)])
            for i in range(n_agents)
        ]
        done_tensors = [
            torch.FloatTensor([done_n[b][i] for b in range(batch_size)])
            for i in range(n_agents)
        ]

        return obs_tensors, act_tensors, rew_tensors, next_obs_tensors, done_tensors

    def __len__(self) -> int:
        return len(self.buffer)
