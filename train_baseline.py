

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from maddpg import MADDPGAgent
from negotiation_env import NegotiationEnv
from replay_buffer import ReplayBuffer


N_EPISODES     = 5000        
BUFFER_SIZE    = 10000     
BATCH_SIZE     = 64        
LR_ACTOR       = 1e-3
LR_CRITIC      = 1e-3
GAMMA          = 0.95      
TAU            = 0.01       
HIDDEN_DIM     = 128
UPDATE_STEPS   = 4          
WARMUP_EPS     = 50         
NOISE_START    = 0.40       
NOISE_END      = 0.02       
NOISE_DECAY    = 3_000      
LOG_EVERY      = 100       
SMOOTH_WINDOW  = 100      


ENV_KWARGS = dict(
    max_rounds      = 30,
    buyer_value     = 1.0,
    seller_cost     = 0.0,
    buyer_discount  = 0.90,   
    seller_discount = 0.90, 
    history_len     = 3,
)

def noise_schedule(episode: int) -> float:
    t = min(episode / NOISE_DECAY, 1.0)
    return NOISE_START + t * (NOISE_END - NOISE_START)


def build_agents(env: NegotiationEnv) -> list[MADDPGAgent]:
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


def run_episode(
    env:     NegotiationEnv,
    agents:  list[MADDPGAgent],
    buffer:  ReplayBuffer,
    noise:   float,
    train:   bool = True,
) -> dict:
    obs, _ = env.reset()
    done   = False

    ep_rewards  = {a: 0.0 for a in env.possible_agents}
    deal_closed = False
    deal_price  = None

    while not done:
        actions = {
            a: agents[i].select_action(obs[a], noise if train else 0.0)
            for i, a in enumerate(env.possible_agents)
        }

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        done = all(terms[a] or truncs[a] for a in env.possible_agents)

        for a in env.possible_agents:
            ep_rewards[a] += rewards[a]
            if infos[a]["deal_closed"]:
                deal_closed = True
                deal_price  = infos[a]["deal_price"]

        if train:
            buffer.add(
                obs_n      = [obs[a]      for a in env.possible_agents],
                act_n      = [actions[a]  for a in env.possible_agents],
                rew_n      = [rewards[a]  for a in env.possible_agents],
                next_obs_n = [next_obs[a] for a in env.possible_agents],
                done_n     = [terms[a] or truncs[a] for a in env.possible_agents],
            )

        obs = next_obs

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


def train(env: NegotiationEnv, agents: list[MADDPGAgent], buffer: ReplayBuffer) -> dict:
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
    history = dict(
        deal_rates    = [],
        deal_prices   = [],
        buyer_rewards = [],
        seller_rewards= [],
        social_welfare= [],
        critic_losses = [],
        actor_losses  = [],
    )


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


def evaluate(env: NegotiationEnv, agents: list[MADDPGAgent], n_eval: int = 500) -> dict:
    buffer = ReplayBuffer(1)    
    results = [
        run_episode(env, agents, buffer, noise=0.0, train=False)
        for _ in range(n_eval)
    ]

    deal_closed  = [r["deal_closed"]          for r in results]
    deal_prices  = [r["deal_price"]           for r in results if r["deal_closed"]]
    welfare      = [r["rewards"]["buyer"] + r["rewards"]["seller"] for r in results]

    return {
        "deal_rate":   np.mean(deal_closed),
        "avg_price":   np.mean(deal_prices) if deal_prices else float("nan"),
        "avg_welfare": np.mean(welfare),
    }

def plot_results(history: dict, env: NegotiationEnv, window: int = SMOOTH_WINDOW):
    n   = len(history["deal_rates"])
    eps = np.arange(n)
    W   = min(window, n)

    def smooth(arr):
        return np.convolve(arr, np.ones(W) / W, mode="valid")

    def eps_sm():
        return eps[W - 1:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MADDPG Baseline - Buyer-Seller Negotiation\n(No Theory of Mind)",
        fontsize=14, fontweight="bold",
    )

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

    ax = axes[0, 1]
    prices_arr = np.array(
        [p if p is not None else np.nan for p in history["deal_prices"]]
    )
    valid = ~np.isnan(prices_arr)
    ax.scatter(eps[valid], prices_arr[valid],
               alpha=0.06, s=4, color="orange", label="Raw deal price")

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

    ax = axes[1, 1]
    ax.plot(eps_sm(), smooth(history["social_welfare"]), lw=2, color="purple")
    ax.set_title("Social Welfare (Buyer + Seller Rewards)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sum of Rewards")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "checkpoints/baseline_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def main(eval_only: bool = False):
    os.makedirs("checkpoints", exist_ok=True)

    env    = NegotiationEnv(**ENV_KWARGS)
    agents = build_agents(env)
    buffer = ReplayBuffer(BUFFER_SIZE)

    if eval_only:
        for i, name in enumerate(env.possible_agents):
            agents[i].load(f"checkpoints/{name}")
    else:
        history = train(env, agents, buffer)
        for i, name in enumerate(env.possible_agents):
            agents[i].save(f"checkpoints/{name}")
        np.save("checkpoints/training_history.npy", history, allow_pickle=True)
        plot_results(history, env)

    evaluate(env, agents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADDPG Baseline Training")
    parser.add_argument("--eval", action="store_true",
                        help="Skip training; load saved models and evaluate")
    args = parser.parse_args()
    main(eval_only=args.eval)
