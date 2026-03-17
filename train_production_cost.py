
from negotiation_env_production_cost import NegotiationEnvProductionCost
from maddpg import MADDPGAgent
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import os

def main(eval_only=False):
    env = NegotiationEnvProductionCost(
        max_rounds=10,
        buyer_value=1.0,
        seller_cost=0.0,
        buyer_discount=0.95,
        seller_discount=0.95,
        history_len=3,
        production_cost=0.2,
        no_deal_penalty=-0.1,
    )

    obs_dim = env.observation_space("buyer").shape[0]
    act_dim = env.action_space("buyer").shape[0]
    obs_total = obs_dim * 2
    act_total = act_dim * 2

    agents = [
        MADDPGAgent(
            agent_idx=0,
            obs_dim=obs_dim,
            act_dim=act_dim,
            obs_dim_total=obs_total,
            act_dim_total=act_total,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.95,
            tau=0.01,
            hidden_dim=64,
        ),
        MADDPGAgent(
            agent_idx=1,
            obs_dim=obs_dim,
            act_dim=act_dim,
            obs_dim_total=obs_total,
            act_dim_total=act_total,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.95,
            tau=0.01,
            hidden_dim=64,
        ),
    ]

    buffer = ReplayBuffer(capacity=1_000_000, obs_dim=obs_dim, act_dim=act_dim)

    episodes = 5_000 if not eval_only else 100
    batch_size = 1024
    update_freq = 100
    eval_freq = 500

    checkpoint_dir = "checkpoints_production_cost"
    os.makedirs(checkpoint_dir, exist_ok=True)
    buyer_path = os.path.join(checkpoint_dir, "buyer")
    seller_path = os.path.join(checkpoint_dir, "seller")
    history_path = os.path.join(checkpoint_dir, "training_history.npy")

    if eval_only:
        agents[0].load(buyer_path)
        agents[1].load(seller_path)
        evaluate(env, agents, episodes=100)
        return

    rewards_history = []
    deal_prices = []
    success_rates = deque(maxlen=100)

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_deal_price = None

        while not done:
            actions = {}
            for i, agent in enumerate(env.agents):
                obs_agent = obs[agent]
                action = agents[i].select_action(obs_agent, noise=0.1 if episode < 4000 else 0.01)
                actions[agent] = action
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = all(terminations.values()) or all(truncations.values())

            buffer.push(obs, actions, rewards, next_obs, done)

            obs = next_obs
            episode_reward += sum(rewards.values()) / len(rewards)

            if infos["buyer"]["deal_closed"]:
                episode_deal_price = infos["buyer"]["deal_price"]

        rewards_history.append(episode_reward)
        if episode_deal_price is not None:
            deal_prices.append(episode_deal_price)
            success_rates.append(1)
        else:
            success_rates.append(0)

        if len(buffer) > batch_size and episode % update_freq == 0:
            for _ in range(update_freq):
                batch = buffer.sample(batch_size)
                for agent in agents:
                    agent.update(batch, agents)

        if episode % eval_freq == 0:
            avg_reward = np.mean(rewards_history[-eval_freq:])
            success_rate = np.mean(success_rates)
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:.3f} | Success Rate: {success_rate:.3f}")

    agents[0].save(buyer_path)
    agents[1].save(seller_path)
    np.save(history_path, np.array(rewards_history))

    evaluate(env, agents, episodes=100)
    plot_results(rewards_history, deal_prices, "production_cost")

def evaluate(env, agents, episodes=100):
    deal_prices = []
    success_count = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            actions = {}
            for i, agent in enumerate(env.agents):
                obs_agent = obs[agent]
                action = agents[i].select_action(obs_agent, noise=0.0)
                actions[agent] = action
            obs, rewards, terminations, truncations, infos = env.step(actions)
            done = all(terminations.values()) or all(truncations.values())

        if infos["buyer"]["deal_closed"]:
            deal_prices.append(infos["buyer"]["deal_price"])
            success_count += 1

    avg_price = np.mean(deal_prices) if deal_prices else 0.0
    success_rate = success_count / episodes
    print(f"Evaluation: Success Rate {success_rate:.3f}, Avg Deal Price {avg_price:.3f}")
    return success_rate, avg_price

def plot_results(rewards_history, deal_prices, suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rewards_history)
    ax1.set_title("Training Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    if deal_prices:
        ax2.hist(deal_prices, bins=20, alpha=0.7)
        ax2.axvline(np.mean(deal_prices), color='red', linestyle='--', label=f'Mean: {np.mean(deal_prices):.3f}')
        ax2.set_title("Deal Price Distribution")
        ax2.set_xlabel("Price")
        ax2.set_ylabel("Frequency")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f"training_results_{suffix}.png")
    plt.show()

if __name__ == "__main__":
    import sys
    eval_only = "--eval" in sys.argv
    main(eval_only=eval_only)
