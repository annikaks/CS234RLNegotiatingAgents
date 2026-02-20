"""
negotiation_env.py
──────────────────
A bilateral buyer-seller negotiation environment built on PettingZoo's
Parallel API.  Both agents act simultaneously every step, making this
a clean fit for MADDPG's joint-training loop.

Game mechanics (inspired by Rubinstein's Alternating-Offers Model):
  - Buyer  wants a LOW price  (max willingness to pay = buyer_value)
  - Seller wants a HIGH price (min acceptable price   = seller_cost)
  - Each round: both agents submit (accept_signal, price_proposal)
  - If either agent accepts the opponent's standing offer → deal closes
  - If neither accepts → proposals become the new standing offers
  - No deal by max_rounds → both receive 0 reward (failure state)

Reward (discounted surplus, à la Rubinstein):
  deal at price p, round t:
    buyer  reward = (buyer_value  - p) * buyer_discount ** t
    seller reward = (p - seller_cost) * seller_discount ** t

Observation (per agent, shape = 4 + history_len):
  [own_reservation, opponent_standing_offer, own_standing_offer,
   time_remaining_norm,  *offer_history]

Action (per agent, shape = 2):
  [accept_signal, price_proposal]
  accept_signal > 0.5  →  accept opponent's current standing offer
  accept_signal ≤ 0.5  →  counter-offer at price_proposal
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces


class NegotiationEnv(ParallelEnv):
    metadata = {"name": "negotiation_v0", "render_modes": ["human"]}

    def __init__(
        self,
        max_rounds: int = 10,
        buyer_value: float = 1.0,
        seller_cost: float = 0.0,
        buyer_discount: float = 0.95,
        seller_discount: float = 0.95,
        history_len: int = 3,
    ):
        """
        Args:
            max_rounds:       Maximum negotiation rounds before deadline (→ 0 reward).
            buyer_value:      Buyer's private value (max willingness to pay). Normalized to [0,1].
            seller_cost:      Seller's reservation cost (min acceptable price).
            buyer_discount:   Buyer's per-round discount factor δ_b  (patience).
            seller_discount:  Seller's per-round discount factor δ_s  (patience).
            history_len:      Number of past mid-offers included in the observation.
        """
        self.max_rounds = max_rounds
        self.buyer_value = buyer_value
        self.seller_cost = seller_cost
        self.buyer_discount = buyer_discount
        self.seller_discount = seller_discount
        self.history_len = history_len

        self.possible_agents = ["buyer", "seller"]

        obs_dim = 4 + history_len   # own_res, opp_offer, own_offer, time_left, history
        act_dim = 2                 # accept_signal, price_proposal

        self._obs_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            for a in self.possible_agents
        }

    # ── PettingZoo required methods ─────────────────────────────────────────────

    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_spaces[agent]

    # ── Core API ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.round = 0

        # Standing offers: seller starts high, buyer starts low
        self.seller_offer = 1.0
        self.buyer_offer  = 0.0

        # Rolling history of mid-points between the two standing offers
        self.history = [0.5] * self.history_len

        self.deal_closed = False
        self.deal_price  = None

        obs   = {a: self._observe(a) for a in self.possible_agents}
        infos = {a: {"deal_closed": False, "deal_price": None}
                 for a in self.possible_agents}
        return obs, infos

    def step(self, actions: dict):
        buyer_act  = actions["buyer"]
        seller_act = actions["seller"]

        buyer_accept    = float(buyer_act[0])  > 0.5
        seller_accept   = float(seller_act[0]) > 0.5
        buyer_proposal  = float(np.clip(buyer_act[1],  0.0, 1.0))
        seller_proposal = float(np.clip(seller_act[1], 0.0, 1.0))

        rewards      = {"buyer": 0.0, "seller": 0.0}
        deal_closed  = False
        deal_price   = None

        # ── Acceptance logic ────────────────────────────────────────────────────
        if buyer_accept and seller_accept:
            # Both accept → split the difference between standing offers
            deal_price  = (self.buyer_offer + self.seller_offer) / 2.0
            deal_closed = True

        elif buyer_accept:
            # Buyer accepts seller's standing offer
            if self.seller_offer <= self.buyer_value:
                deal_price  = self.seller_offer
                deal_closed = True

        elif seller_accept:
            # Seller accepts buyer's standing offer
            if self.buyer_offer >= self.seller_cost:
                deal_price  = self.buyer_offer
                deal_closed = True

        # ── Implicit acceptance: proposals already overlap ───────────────────────
        elif buyer_proposal >= seller_proposal:
            deal_price  = (buyer_proposal + seller_proposal) / 2.0
            deal_closed = True

        if deal_closed:
            self.deal_closed = True
            self.deal_price  = deal_price
            t = self.round
            rewards["buyer"]  = (self.buyer_value - deal_price)  * (self.buyer_discount  ** t)
            rewards["seller"] = (deal_price - self.seller_cost)  * (self.seller_discount ** t)

        else:
            # Update standing offers and history
            self.buyer_offer  = buyer_proposal
            self.seller_offer = seller_proposal
            mid = (buyer_proposal + seller_proposal) / 2.0
            self.history.append(mid)
            self.round += 1

        # ── Deadline check ───────────────────────────────────────────────────────
        done = deal_closed or (self.round >= self.max_rounds)

        terminations = {a: done for a in self.possible_agents}
        truncations  = {a: False for a in self.possible_agents}
        infos        = {
            a: {"deal_closed": deal_closed, "deal_price": deal_price}
            for a in self.possible_agents
        }

        # PettingZoo convention: clear agents list when done
        if done:
            self.agents = []

        obs = {a: self._observe(a) for a in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def render(self):
        status = (f"DEAL @ {self.deal_price:.3f}" if self.deal_closed
                  else f"Buyer={self.buyer_offer:.3f}  Seller={self.seller_offer:.3f}")
        print(f"[Round {self.round:2d}/{self.max_rounds}]  {status}")

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _observe(self, agent: str) -> np.ndarray:
        remaining = 1.0 - (self.round / self.max_rounds)
        hist      = self.history[-self.history_len:]
        # Pad if game just started (shouldn't happen since we pre-fill)
        if len(hist) < self.history_len:
            hist = [0.5] * (self.history_len - len(hist)) + list(hist)

        if agent == "buyer":
            own_res   = self.buyer_value
            opp_offer = self.seller_offer
            own_offer = self.buyer_offer
        else:
            own_res   = self.seller_cost
            opp_offer = self.buyer_offer
            own_offer = self.seller_offer

        return np.array(
            [own_res, opp_offer, own_offer, remaining] + list(hist),
            dtype=np.float32,
        )

    # ── Rubinstein benchmark (for reference / evaluation) ────────────────────────

    def rubinstein_equilibrium(self) -> dict:
        """
        Computes the Rubinstein alternating-offers subgame perfect equilibrium.
        Assumes seller makes the first offer.

        Returns deal price and share for each agent.
        Note: this is the ALTERNATING-OFFERS equilibrium, used as a benchmark
        even though our env uses simultaneous proposals.
        """
        db, ds = self.buyer_discount, self.seller_discount
        # Seller's equilibrium share when making first offer
        seller_share = (1 - db) / (1 - db * ds)
        deal_price   = self.seller_cost + seller_share * (self.buyer_value - self.seller_cost)
        return {
            "deal_price":   deal_price,
            "seller_share": seller_share,
            "buyer_share":  1 - seller_share,
        }

    def nash_bargaining_solution(self) -> dict:
        """
        Nash Bargaining Solution (equal-split of surplus when δ_b == δ_s).
        """
        mid = (self.buyer_value + self.seller_cost) / 2.0
        return {
            "deal_price":   mid,
            "buyer_share":  0.5,
            "seller_share": 0.5,
        }
