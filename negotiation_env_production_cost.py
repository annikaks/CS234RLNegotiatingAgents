"""
negotiation_env_production_cost.py
──────────────────────────────────
Experiment-specific variant of the bilateral buyer-seller negotiation
environment. This keeps production-cost reward shaping isolated from the
baseline environment used by the rest of the project.
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class NegotiationEnvProductionCost(ParallelEnv):
    metadata = {"name": "negotiation_production_cost_v0", "render_modes": ["human"]}

    def __init__(
        self,
        max_rounds: int = 10,
        buyer_value: float = 1.0,
        seller_cost: float = 0.0,
        buyer_discount: float = 0.95,
        seller_discount: float = 0.95,
        history_len: int = 3,
        production_cost: float = 0.0,
        no_deal_penalty: float = 0.0,
    ):
        self.max_rounds = max_rounds
        self.buyer_value = buyer_value
        self.seller_cost = seller_cost
        self.buyer_discount = buyer_discount
        self.seller_discount = seller_discount
        self.history_len = history_len
        self.production_cost = production_cost
        self.no_deal_penalty = no_deal_penalty

        self.possible_agents = ["buyer", "seller"]

        obs_dim = 4 + history_len
        act_dim = 2

        self._obs_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.round = 0
        self.seller_offer = 1.0
        self.buyer_offer = 0.0
        self.history = [0.5] * self.history_len
        self.deal_closed = False
        self.deal_price = None

        obs = {a: self._observe(a) for a in self.possible_agents}
        infos = {
            a: {"deal_closed": False, "deal_price": None}
            for a in self.possible_agents
        }
        return obs, infos

    def step(self, actions: dict):
        buyer_act = actions["buyer"]
        seller_act = actions["seller"]

        buyer_accept = float(buyer_act[0]) > 0.5
        seller_accept = float(seller_act[0]) > 0.5
        buyer_proposal = float(np.clip(buyer_act[1], 0.0, 1.0))
        seller_proposal = float(np.clip(seller_act[1], 0.0, 1.0))

        rewards = {"buyer": 0.0, "seller": 0.0}
        deal_closed = False
        deal_price = None

        if buyer_accept and seller_accept:
            deal_price = (self.buyer_offer + self.seller_offer) / 2.0
            deal_closed = True
        elif buyer_accept:
            if self.seller_offer <= self.buyer_value:
                deal_price = self.seller_offer
                deal_closed = True
        elif seller_accept:
            if self.buyer_offer >= self.seller_cost:
                deal_price = self.buyer_offer
                deal_closed = True
        elif buyer_proposal >= seller_proposal:
            deal_price = (buyer_proposal + seller_proposal) / 2.0
            deal_closed = True

        if deal_closed:
            self.deal_closed = True
            self.deal_price = deal_price
            t = self.round
            rewards["buyer"] = (self.buyer_value - deal_price) * (self.buyer_discount ** t)
            rewards["seller"] = (deal_price - self.production_cost) * (self.seller_discount ** t)
        else:
            self.buyer_offer = buyer_proposal
            self.seller_offer = seller_proposal
            mid = (buyer_proposal + seller_proposal) / 2.0
            self.history.append(mid)
            self.round += 1

        done = deal_closed or (self.round >= self.max_rounds)

        if done and not deal_closed:
            rewards["buyer"] = self.no_deal_penalty
            rewards["seller"] = self.no_deal_penalty

        terminations = {a: done for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        infos = {
            a: {"deal_closed": deal_closed, "deal_price": deal_price}
            for a in self.possible_agents
        }

        if done:
            self.agents = []

        obs = {a: self._observe(a) for a in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        status = (
            f"DEAL @ {self.deal_price:.3f}"
            if self.deal_closed
            else f"Buyer={self.buyer_offer:.3f}  Seller={self.seller_offer:.3f}"
        )
        print(f"[Round {self.round:2d}/{self.max_rounds}]  {status}")

    def _observe(self, agent: str) -> np.ndarray:
        remaining = 1.0 - (self.round / self.max_rounds)
        hist = self.history[-self.history_len:]
        if len(hist) < self.history_len:
            hist = [0.5] * (self.history_len - len(hist)) + list(hist)

        if agent == "buyer":
            own_res = self.buyer_value
            opp_offer = self.seller_offer
            own_offer = self.buyer_offer
        else:
            own_res = self.production_cost
            opp_offer = self.buyer_offer
            own_offer = self.seller_offer

        return np.array(
            [own_res, opp_offer, own_offer, remaining] + list(hist),
            dtype=np.float32,
        )

    def rubinstein_equilibrium(self) -> dict:
        db, ds = self.buyer_discount, self.seller_discount
        seller_share = (1 - db) / (1 - db * ds)
        deal_price = self.seller_cost + seller_share * (self.buyer_value - self.seller_cost)
        return {
            "deal_price": deal_price,
            "seller_share": seller_share,
            "buyer_share": 1 - seller_share,
        }

    def nash_bargaining_solution(self) -> dict:
        mid = (self.buyer_value + self.seller_cost) / 2.0
        return {
            "deal_price": mid,
            "buyer_share": 0.5,
            "seller_share": 0.5,
        }
