"""
train_baseline.py
─────────────────
MADDPG baseline training for the buyer-seller negotiation environment.
No Theory of Mind – agents learn purely through self-play.

Usage:
    python train_baseline.py           # trains with default config
    python train_baseline.py --eval    # loads saved agents and runs evaluation only

Outputs (written to ./checkpoints/):
    buyer_actor.pt / buyer_critic.pt
    seller_actor.pt / seller_critic.pt
    training_history.npy               (dict of per-episode metrics)
    baseline_results.png               (4-panel diagnostic plot)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from maddpg import MADDPGAgent
from negotiation_env import NegotiationEnv
from replay_buffer import ReplayBuffer

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_EPISODES     = 3000        # total training episodes
BUFFER_SIZE    = 10000     # replay buffer capacity
BATCH_SIZE     = 64         # mini-batch size for gradient updates
LR_ACTOR       = 1e-3
LR_CRITIC      = 1e-3
GAMMA          = 0.95       # RL discount (separate from negotiation discount)
TAU            = 0.01       # Polyak soft-update coefficient
HIDDEN_DIM     = 128
UPDATE_STEPS   = 4          # gradient updates per episode
WARMUP_EPS     = 50         # random-action episodes to pre-fill buffer
NOISE_START    = 0.40       # initial Gaussian exploration std
NOISE_END      = 0.02       # final   Gaussian exploration std
NOISE_DECAY    = 3_000      # episodes over which noise anneals
LOG_EVERY      = 100        # console log frequency (episodes)
SMOOTH_WINDOW  = 100        # moving-average window for plots

# ── Environment config ─────────────────────────────────────────────────────────

ENV_KWARGS = dict(
    max_rounds      = 10,
    buyer_value     = 1.0,
    seller_cost     = 0.0,
    buyer_discount  = 0.95,   # δ_b  (patience; higher = more patient)
    seller_discount = 0.95,   # δ_s
    history_len     = 3,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def noise_schedule(episode: int) -> float:
    """Linear annealing from NOISE_START → NOISE_END over NOISE_DECAY episodes."""
    t = min(episode / NOISE_DECAY, 1.0)
    return NOISE_START + t * (NOISE_END - NOISE_START)


def build_agents(env: NegotiationEnv) -> list[MADDPGAgent]:
    """Instantiate one MADDPGAgent per environment agent."""
    obs_dims  = [env.observation_space(a).shape[0] for a in env.possible_agents]
    act_dims  = [env.action_space(a).shape[0]      for a in env.possible_agents]
    obs_total = sum(obs_dims)
    act_total = sum(act_dims)

    return [
        MADDPGAgent(
            agent_idx     = i,
            obs_dim       = obs_dims[i],
            act_dim       = act_dims[i],
            obs_dim_total = obs_total,
            act_dim_total = act_total,
            lr_actor      = LR_ACTOR,
            lr_critic     = LR_CRITIC,
            gamma         = GAMMA,
            tau           = TAU,
            hidden_dim    = HIDDEN_DIM,
        )
        for i in range(len(env.possible_agents))
    ]


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    env:     NegotiationEnv,
    agents:  list[MADDPGAgent],
    buffer:  ReplayBuffer,
    noise:   float,
    train:   bool = True,
) -> dict:
    """
    Run one full episode.

    Returns a dict with:
        rewards      – {agent_name: cumulative reward}
        deal_closed  – bool
        deal_price   – float | None
        critic_loss  – mean critic loss this episode (NaN if no update)
        actor_loss   – mean actor  loss this episode
    """
    obs, _ = env.reset()
    done   = False

    ep_rewards  = {a: 0.0 for a in env.possible_agents}
    deal_closed = False
    deal_price  = None

    while not done:
        # ── Select actions ──────────────────────────────────────────────────
        actions = {
            a: agents[i].select_action(obs[a], noise if train else 0.0)
            for i, a in enumerate(env.possible_agents)
        }

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        done = all(terms[a] or truncs[a] for a in env.possible_agents)

        # ── Accumulate episode stats ────────────────────────────────────────
        for a in env.possible_agents:
            ep_rewards[a] += rewards[a]
            if infos[a]["deal_closed"]:
                deal_closed = True
                deal_price  = infos[a]["deal_price"]

        # ── Store joint transition ──────────────────────────────────────────
        if train:
            buffer.add(
                obs_n      = [obs[a]      for a in env.possible_agents],
                act_n      = [actions[a]  for a in env.possible_agents],
                rew_n      = [rewards[a]  for a in env.possible_agents],
                next_obs_n = [next_obs[a] for a in env.possible_agents],
                done_n     = [terms[a] or truncs[a] for a in env.possible_agents],
            )

        obs = next_obs

    # ── Gradient updates ─────────────────────────────────────────────────────
    c_losses, a_losses = [], []
    if train and len(buffer) >= BATCH_SIZE:
        for _ in range(UPDATE_STEPS):
            batch = buffer.sample(BATCH_SIZE)
            for agent in agents:
                c, a = agent.update(batch, agents)
                c_losses.append(c)
                a_losses.append(a)

    return dict(
        rewards     = ep_rewards,
        deal_closed = deal_closed,
        deal_price  = deal_price,
        critic_loss = float(np.mean(c_losses)) if c_losses else float("nan"),
        actor_loss  = float(np.mean(a_losses)) if a_losses else float("nan"),
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train(env: NegotiationEnv, agents: list[MADDPGAgent], buffer: ReplayBuffer) -> dict:
    # ── Warm-up: fill buffer with random transitions ─────────────────────────
    print(f"\nWarm-up: {WARMUP_EPS} random episodes …")
    for _ in range(WARMUP_EPS):
        obs, _ = env.reset()
        done   = False
        while not done:
            actions  = {a: env.action_space(a).sample() for a in env.possible_agents}
            nobs, rw, terms, truncs, _ = env.step(actions)
            done = all(terms[a] or truncs[a] for a in env.possible_agents)
            buffer.add(
                obs_n      = [obs[a]     for a in env.possible_agents],
                act_n      = [actions[a] for a in env.possible_agents],
                rew_n      = [rw[a]      for a in env.possible_agents],
                next_obs_n = [nobs[a]    for a in env.possible_agents],
                done_n     = [terms[a] or truncs[a] for a in env.possible_agents],
            )
            obs = nobs
    print(f"Warm-up done.  Buffer size: {len(buffer):,}\n")

    # ── Metrics ───────────────────────────────────────────────────────────────
    history = dict(
        deal_rates    = [],
        deal_prices   = [],
        buyer_rewards = [],
        seller_rewards= [],
        social_welfare= [],
        critic_losses = [],
        actor_losses  = [],
    )

    print("=" * 65)
    print("  MADDPG Baseline – Buyer-Seller Negotiation")
    print("  (No Theory of Mind)")
    print("=" * 65)
    print(f"  {'Episodes':12s}: {N_EPISODES}")
    print(f"  {'Batch size':12s}: {BATCH_SIZE}")
    print(f"  {'Buffer size':12s}: {BUFFER_SIZE:,}")
    print(f"  {'δ_buyer':12s}: {ENV_KWARGS['buyer_discount']}"
          f"    {'δ_seller':12s}: {ENV_KWARGS['seller_discount']}")
    print("=" * 65 + "\n")

    for ep in range(N_EPISODES):
        noise  = noise_schedule(ep)
        result = run_episode(env, agents, buffer, noise, train=True)

        history["deal_rates"].append(1.0 if result["deal_closed"] else 0.0)
        history["deal_prices"].append(result["deal_price"])
        history["buyer_rewards"].append(result["rewards"]["buyer"])
        history["seller_rewards"].append(result["rewards"]["seller"])
        history["social_welfare"].append(
            result["rewards"]["buyer"] + result["rewards"]["seller"]
        )
        history["critic_losses"].append(result["critic_loss"])
        history["actor_losses"].append(result["actor_loss"])

        if (ep + 1) % LOG_EVERY == 0:
            w           = min(LOG_EVERY, ep + 1)
            deal_rate   = np.mean(history["deal_rates"][-w:])
            raw_prices  = [p for p in history["deal_prices"][-w:] if p is not None]
            avg_price   = np.mean(raw_prices) if raw_prices else float("nan")
            welfare     = np.mean(history["social_welfare"][-w:])
            print(
                f"Ep {ep+1:5d} | Deal%: {deal_rate*100:5.1f}% | "
                f"Price: {avg_price:.3f} | Welfare: {welfare:.4f} | "
                f"Noise: {noise:.3f}"
            )

    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(env: NegotiationEnv, agents: list[MADDPGAgent], n_eval: int = 500) -> dict:
    """Run n_eval deterministic episodes and report aggregate statistics."""
    buffer = ReplayBuffer(1)    # dummy buffer (not used)
    results = [
        run_episode(env, agents, buffer, noise=0.0, train=False)
        for _ in range(n_eval)
    ]

    deal_closed  = [r["deal_closed"]          for r in results]
    deal_prices  = [r["deal_price"]           for r in results if r["deal_closed"]]
    welfare      = [r["rewards"]["buyer"] + r["rewards"]["seller"] for r in results]

    print("\n── Evaluation Results ──────────────────────────────────────")
    print(f"  Episodes evaluated : {n_eval}")
    print(f"  Deal closure rate  : {np.mean(deal_closed)*100:.1f}%")
    if deal_prices:
        print(f"  Avg deal price     : {np.mean(deal_prices):.4f}  "
              f"(std={np.std(deal_prices):.4f})")
    print(f"  Avg social welfare : {np.mean(welfare):.4f}")

    rubinstein = env.rubinstein_equilibrium()
    nash       = env.nash_bargaining_solution()
    print(f"\n  Rubinstein eq. price : {rubinstein['deal_price']:.4f}")
    print(f"  Nash bargaining price: {nash['deal_price']:.4f}")
    print("────────────────────────────────────────────────────────────\n")
    return {
        "deal_rate":   np.mean(deal_closed),
        "avg_price":   np.mean(deal_prices) if deal_prices else float("nan"),
        "avg_welfare": np.mean(welfare),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(history: dict, env: NegotiationEnv, window: int = SMOOTH_WINDOW):
    """Generate the 4-panel diagnostic figure and save it."""
    n   = len(history["deal_rates"])
    eps = np.arange(n)
    W   = min(window, n)

    def smooth(arr):
        return np.convolve(arr, np.ones(W) / W, mode="valid")

    def eps_sm():
        return eps[W - 1:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MADDPG Baseline – Buyer-Seller Negotiation\n(No Theory of Mind)",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: Deal Closure Rate ───────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(eps_sm(), smooth(history["deal_rates"]), lw=2, color="steelblue",
            label="Deal rate")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.4, label="100%")
    ax.set_ylim(0, 1.1)
    ax.set_title("Deal Closure Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 2: Deal Price vs Theoretical Benchmarks ────────────────────────
    ax = axes[0, 1]
    prices_arr = np.array(
        [p if p is not None else np.nan for p in history["deal_prices"]]
    )
    valid = ~np.isnan(prices_arr)
    ax.scatter(eps[valid], prices_arr[valid],
               alpha=0.06, s=4, color="orange", label="Raw deal price")

    # Smoothed (NaN-aware)
    price_ma = [
        np.nanmean(prices_arr[max(0, i - W + 1): i + 1])
        for i in range(W - 1, n)
    ]
    ax.plot(eps_sm(), price_ma, lw=2, color="darkorange", label="Moving avg")

    rubinstein = env.rubinstein_equilibrium()
    nash       = env.nash_bargaining_solution()
    ax.axhline(nash["deal_price"],       ls="--", color="green", lw=1.5,
               label=f"Nash (={nash['deal_price']:.2f})")
    ax.axhline(rubinstein["deal_price"], ls="--", color="red",   lw=1.5,
               label=f"Rubinstein (={rubinstein['deal_price']:.3f})")

    ax.set_ylim(0, 1)
    ax.set_title("Deal Price vs Theoretical Benchmarks")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 3: Per-Agent Rewards ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(eps_sm(), smooth(history["buyer_rewards"]),  lw=2,
            color="royalblue", label="Buyer")
    ax.plot(eps_sm(), smooth(history["seller_rewards"]), lw=2,
            color="crimson",   label="Seller")
    ax.set_title("Agent Rewards (Moving Avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Discounted Reward")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 4: Social Welfare ───────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(eps_sm(), smooth(history["social_welfare"]), lw=2, color="purple")
    ax.set_title("Social Welfare  (Buyer + Seller Rewards)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sum of Rewards")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "checkpoints/baseline_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(eval_only: bool = False):
    os.makedirs("checkpoints", exist_ok=True)

    env    = NegotiationEnv(**ENV_KWARGS)
    agents = build_agents(env)
    buffer = ReplayBuffer(BUFFER_SIZE)

    if eval_only:
        print("Loading saved agents …")
        for i, name in enumerate(env.possible_agents):
            agents[i].load(f"checkpoints/{name}")
    else:
        history = train(env, agents, buffer)

        # ── Save ──────────────────────────────────────────────────────────────
        for i, name in enumerate(env.possible_agents):
            agents[i].save(f"checkpoints/{name}")
        np.save("checkpoints/training_history.npy", history, allow_pickle=True)
        print("\nCheckpoints saved to ./checkpoints/")

        plot_results(history, env)

    evaluate(env, agents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADDPG Baseline Training")
    parser.add_argument("--eval", action="store_true",
                        help="Skip training; load saved models and evaluate")
    args = parser.parse_args()
    main(eval_only=args.eval)
