"""
replay_buffer.py
────────────────
Experience replay buffer for multi-agent transitions.

Each stored transition is a joint tuple:
    (obs_n, act_n, rew_n, next_obs_n, done_n)

where the *_n suffix denotes a list with one element per agent.

Sample returns PyTorch tensors, one per agent, shaped [batch_size, dim].
"""

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store.
                      Oldest entries are evicted once full (deque behaviour).
        """
        self.buffer = deque(maxlen=capacity)

    # ── Public API ───────────────────────────────────────────────────────────────

    def add(
        self,
        obs_n:      list,   # list[np.ndarray]  – one per agent
        act_n:      list,   # list[np.ndarray]
        rew_n:      list,   # list[float]
        next_obs_n: list,   # list[np.ndarray]
        done_n:     list,   # list[bool]
    ):
        """Store one joint transition."""
        self.buffer.append((
            [np.array(o, dtype=np.float32) for o in obs_n],
            [np.array(a, dtype=np.float32) for a in act_n],
            [float(r)                      for r in rew_n],
            [np.array(o, dtype=np.float32) for o in next_obs_n],
            [float(d)                      for d in done_n],
        ))

    def sample(self, batch_size: int):
        """
        Sample a random mini-batch.

        Returns a tuple of lists, one list per field (obs, act, rew, next_obs, done).
        Each list contains n_agents tensors of shape [batch_size, dim].
        """
        batch    = random.sample(self.buffer, batch_size)
        obs_n, act_n, rew_n, next_obs_n, done_n = zip(*batch)

        n_agents = len(obs_n[0])

        def _to_tensor_list(data, squeeze=False):
            """Stack batch dimension per agent, convert to FloatTensor."""
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

        # Rewards and dones are scalars per agent → shape [batch_size]
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
