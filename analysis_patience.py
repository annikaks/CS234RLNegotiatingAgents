import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from maddpg import MADDPGAgent
from negotiation_env import NegotiationEnv
from replay_buffer import ReplayBuffer



SYMMETRIC_DELTAS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
ASYM_DELTAS      = [0.50, 0.70, 0.85, 0.95]          



N_EPISODES   = 2000
WARMUP_EPS   = 30
BUFFER_SIZE  = 10_000
BATCH_SIZE   = 64
LR_ACTOR     = 1e-3
LR_CRITIC    = 1e-3
GAMMA        = 0.95
TAU          = 0.01
HIDDEN_DIM   = 128
UPDATE_STEPS = 4
NOISE_START  = 0.40
NOISE_END    = 0.02
NOISE_DECAY  = 5_000
N_EVAL       = 300       

BASE_ENV_KWARGS = dict(
    max_rounds  = 10,
    buyer_value = 1.0,
    seller_cost = 0.0,
    history_len = 3,
)

OUT_DIR = "checkpoints/patience_analysis"

def _noise_schedule(ep: int) -> float:
    t = min(ep / NOISE_DECAY, 1.0)
    return NOISE_START + t * (NOISE_END - NOISE_START)


def _build_agents(env: NegotiationEnv) -> list:
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


def _run_episode(env, agents, buffer, noise, train=True) -> dict:
    obs, _      = env.reset()
    done        = False
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

    if train and len(buffer) >= BATCH_SIZE:
        for _ in range(UPDATE_STEPS):
            batch = buffer.sample(BATCH_SIZE)
            for agent in agents:
                agent.update(batch, agents)

    return dict(rewards=ep_rewards, deal_closed=deal_closed, deal_price=deal_price)


def _warmup(env, buffer):
    for _ in range(WARMUP_EPS):
        obs, _ = env.reset()
        done   = False
        while not done:
            actions = {a: env.action_space(a).sample() for a in env.possible_agents}
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




def train_and_evaluate(buyer_discount: float, seller_discount: float) -> dict:
 
    env_kwargs = {
        **BASE_ENV_KWARGS,
        "buyer_discount":  buyer_discount,
        "seller_discount": seller_discount,
    }
    env    = NegotiationEnv(**env_kwargs)
    agents = _build_agents(env)
    buffer = ReplayBuffer(BUFFER_SIZE)

    _warmup(env, buffer)

    for ep in range(N_EPISODES):
        _run_episode(env, agents, buffer, _noise_schedule(ep), train=True)


    dummy_buf        = ReplayBuffer(1)
    deal_flags       = []
    deal_prices_list = []
    welfare_list     = []
    buyer_rew_list   = []
    seller_rew_list  = []

    for _ in range(N_EVAL):
        r = _run_episode(env, agents, dummy_buf, noise=0.0, train=False)
        deal_flags.append(r["deal_closed"])
        if r["deal_closed"]:
            deal_prices_list.append(r["deal_price"])
        welfare_list.append(r["rewards"]["buyer"] + r["rewards"]["seller"])
        buyer_rew_list.append(r["rewards"]["buyer"])
        seller_rew_list.append(r["rewards"]["seller"])

    rubinstein = env.rubinstein_equilibrium()
    nash       = env.nash_bargaining_solution()
    avg_price  = float(np.mean(deal_prices_list)) if deal_prices_list else float("nan")

    return {
        "buyer_discount":    buyer_discount,
        "seller_discount":   seller_discount,
        "deal_rate":         float(np.mean(deal_flags)),
        "avg_deal_price":    avg_price,
        "avg_welfare":       float(np.mean(welfare_list)),
        "avg_buyer_reward":  float(np.mean(buyer_rew_list)),
        "avg_seller_reward": float(np.mean(seller_rew_list)),
        "rubinstein_price":  rubinstein["deal_price"],
        "nash_price":        nash["deal_price"],
        "price_error":       abs(avg_price - rubinstein["deal_price"])
                             if not np.isnan(avg_price) else float("nan"),
    }



def run_symmetric_sweep() -> list[dict]:

    
    results = []
    for δ in SYMMETRIC_DELTAS:
        print(f"  δ = {δ:.2f}  …", end="", flush=True)
        r = train_and_evaluate(δ, δ)
        results.append(r)
        print(
            f"  deal_rate={r['deal_rate']*100:.1f}%  "
            f"price={r['avg_deal_price']:.3f}  "
            f"(Rubinstein={r['rubinstein_price']:.3f})"
        )
    return results


def run_asymmetric_sweep() -> list[dict]:

    results = []
    total = len(ASYM_DELTAS) ** 2
    done  = 0
    for db in ASYM_DELTAS:
        for ds in ASYM_DELTAS:
            done += 1
            print(f"  [{done:2d}/{total}]  δ_b={db:.2f}  δ_s={ds:.2f}  …",
                  end="", flush=True)
            r = train_and_evaluate(db, ds)
            results.append(r)
            print(
                f"  price={r['avg_deal_price']:.3f}  "
                f"Rub={r['rubinstein_price']:.3f}  "
                f"deal%={r['deal_rate']*100:.1f}%"
            )
    return results




def plot_symmetric(results: list[dict]):
  
    deltas      = [r["buyer_discount"]   for r in results]
    deal_rates  = [r["deal_rate"]        for r in results]
    avg_prices  = [r["avg_deal_price"]   for r in results]
    rub_prices  = [r["rubinstein_price"] for r in results]
    nash_prices = [r["nash_price"]       for r in results]
    welfares    = [r["avg_welfare"]      for r in results]
    b_rews      = [r["avg_buyer_reward"] for r in results]
    s_rews      = [r["avg_seller_reward"]for r in results]
    errors      = [r["price_error"]      for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Effect of Patience (δ) on Negotiation Outcomes\n"
        "Symmetric Case: δ_buyer = δ_seller = δ",
        fontsize=14, fontweight="bold",
    )


    ax = axes[0, 0]
    ax.plot(deltas, deal_rates, "o-", lw=2, color="steelblue", ms=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Deal Closure Rate vs δ")
    ax.set_xlabel("Patience factor δ")
    ax.set_ylabel("Deal rate")
    ax.grid(alpha=0.3)
    for x, y in zip(deltas, deal_rates):
        ax.annotate(f"{y*100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)


    ax = axes[0, 1]
    ax.plot(deltas, avg_prices,  "o-",  lw=2, color="darkorange",
            ms=8,  label="MADDPG learned price")
    ax.plot(deltas, rub_prices,  "s--", lw=1.5, color="red",
            ms=7,  label="Rubinstein eq. price")
    ax.plot(deltas, nash_prices, "^:",  lw=1.5, color="green",
            ms=7,  label="Nash bargaining price")
    ax.set_ylim(0, 1)
    ax.set_title("Deal Price vs Theoretical Benchmarks")
    ax.set_xlabel("Patience factor δ")
    ax.set_ylabel("Deal price  (0=seller cost, 1=buyer value)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


    ax = axes[1, 0]
    ax.plot(deltas, b_rews, "o-", lw=2, color="royalblue", ms=8, label="Buyer reward")
    ax.plot(deltas, s_rews, "s-", lw=2, color="crimson",   ms=8, label="Seller reward")
    ax.plot(deltas, welfares, "^-", lw=2, color="purple", ms=8,
            label="Social welfare (sum)", linestyle="dashed")
    ax.set_title("Agent Rewards and Social Welfare vs δ")
    ax.set_xlabel("Patience factor δ")
    ax.set_ylabel("Discounted reward")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

   
    ax = axes[1, 1]
    ax.bar(
        [str(d) for d in deltas], errors,
        color=["#d73027" if e > 0.1 else "#fdae61" if e > 0.05 else "#1a9850"
               for e in errors],
        edgecolor="black", linewidth=0.6,
    )
    ax.set_title("|Learned Price - Rubinstein Price|")
    ax.set_xlabel("Patience factor δ")
    ax.set_ylabel("Absolute deviation")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(errors) * 1.4 if any(not np.isnan(e) for e in errors) else 0.3)
    for i, (x, e) in enumerate(zip(range(len(deltas)), errors)):
        if not np.isnan(e):
            ax.text(i, e + 0.002, f"{e:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "patience_symmetric.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSymmetric plot saved → {path}")


def plot_heatmaps(results: list[dict]):

    n   = len(ASYM_DELTAS)
    idx = {d: i for i, d in enumerate(ASYM_DELTAS)}


    deal_rate_grid   = np.full((n, n), np.nan)
    price_grid       = np.full((n, n), np.nan)
    welfare_grid     = np.full((n, n), np.nan)
    rub_grid         = np.full((n, n), np.nan)  
    price_error_grid = np.full((n, n), np.nan)

    for r in results:
        i = idx[r["seller_discount"]]   
        j = idx[r["buyer_discount"]]    
        deal_rate_grid[i, j]   = r["deal_rate"]
        price_grid[i, j]       = r["avg_deal_price"]
        welfare_grid[i, j]     = r["avg_welfare"]
        rub_grid[i, j]         = r["rubinstein_price"]
        price_error_grid[i, j] = r["price_error"]

    tick_labels = [str(d) for d in ASYM_DELTAS]

    def _annotate(ax, data):
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                txt = f"{v:.2f}" if not np.isnan(v) else "—"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color="white",
                        fontweight="bold")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Effect of Patience on Negotiation Outcomes — Asymmetric Analysis\n"
        "Rows = δ_seller,  Columns = δ_buyer",
        fontsize=13, fontweight="bold",
    )

    def _setup_ax(ax, title, xlabel="δ_buyer", ylabel="δ_seller"):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels)
        ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels)

    ax = axes[0, 0]
    im = ax.imshow(deal_rate_grid, vmin=0, vmax=1,
                   cmap="RdYlGn", origin="lower")
    _annotate(ax, deal_rate_grid)
    _setup_ax(ax, "Deal Closure Rate")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(price_grid, vmin=0, vmax=1,
                   cmap="coolwarm", origin="lower")
    _annotate(ax, price_grid)
    _setup_ax(ax, "Learned Deal Price\n(blue=low/buyer-favored, red=high/seller-favored)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    try:
        cs = axes[0, 1].contour(
            rub_grid, levels=5, colors="black", linewidths=0.8, linestyles="--",
            origin="lower",
        )
        axes[0, 1].clabel(cs, inline=True, fontsize=7, fmt="Rub=%.2f")
    except Exception:
        pass

    ax = axes[1, 0]
    im = ax.imshow(welfare_grid, cmap="YlOrRd", origin="lower")
    _annotate(ax, welfare_grid)
    _setup_ax(ax, "Social Welfare (Buyer + Seller Reward)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    ax = axes[1, 1]
    im = ax.imshow(price_error_grid, vmin=0, cmap="Reds", origin="lower")
    _annotate(ax, price_error_grid)
    _setup_ax(ax, "|Learned Price - Rubinstein Price|")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "patience_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap plot saved  → {path}")


def plot_asymmetric_lines(results: list[dict]):
    import matplotlib.lines as mlines

    colors  = {0.50: "steelblue", 0.70: "darkorange", 0.85: "crimson", 0.95: "purple"}
    markers = {0.50: "o",         0.70: "s",           0.85: "^",       0.95: "D"}

    from collections import defaultdict
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["seller_discount"]].append(r)
    for ds in by_ds:
        by_ds[ds].sort(key=lambda r: r["buyer_discount"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Effect of Patience (δ) on Negotiation Outcomes\n"
        "Asymmetric Case: each line = fixed δ_seller,  x-axis = δ_buyer",
        fontsize=14, fontweight="bold",
    )


    all_pairs  = sorted(results, key=lambda r: (r["buyer_discount"], r["seller_discount"]))
    bar_labels = [f"b={r['buyer_discount']:.2f}\ns={r['seller_discount']:.2f}" for r in all_pairs]
    bar_errors = [r["price_error"] for r in all_pairs]
    bar_colors = ["#d73027" if e > 0.1 else "#fdae61" if e > 0.05 else "#1a9850"
                  for e in bar_errors]

    for ds, group in sorted(by_ds.items()):
        dbs         = [r["buyer_discount"]    for r in group]
        prices      = [r["avg_deal_price"]    for r in group]
        rub_prices  = [r["rubinstein_price"]  for r in group]
        nash_prices = [r["nash_price"]        for r in group]
        deal_rates  = [r["deal_rate"]         for r in group]
        welfares    = [r["avg_welfare"]       for r in group]
        b_rews      = [r["avg_buyer_reward"]  for r in group]
        s_rews      = [r["avg_seller_reward"] for r in group]
        c, m = colors[ds], markers[ds]

      
        axes[0, 0].plot(dbs, deal_rates, f"{m}-", color=c, lw=2, ms=8,
                        label=f"δ_s={ds:.2f}")
        for x, y in zip(dbs, deal_rates):
            axes[0, 0].annotate(f"{y*100:.0f}%", (x, y),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=7)


        axes[0, 1].plot(dbs, prices,      f"{m}-",  color=c, lw=2,   ms=8)
        axes[0, 1].plot(dbs, rub_prices,  f"{m}--", color=c, lw=1.5, ms=5, alpha=0.6)
        axes[0, 1].plot(dbs, nash_prices, f"{m}:",  color=c, lw=1.5, ms=5, alpha=0.6)


        axes[1, 0].plot(dbs, b_rews,   f"{m}-",  color=c, lw=2,   ms=8)
        axes[1, 0].plot(dbs, s_rews,   f"{m}:",  color=c, lw=2,   ms=8)
        axes[1, 0].plot(dbs, welfares, f"{m}--", color=c, lw=1.5, ms=6, alpha=0.7)


    color_handles = [
        mlines.Line2D([], [], color=colors[ds], marker=markers[ds], ms=7, lw=2,
                      label=f"δ_s={ds:.2f}")
        for ds in sorted(colors)
    ]


    ax = axes[0, 0]
    ax.set_ylim(0, 1.15)
    ax.set_title("Deal Closure Rate vs δ_buyer")
    ax.set_xlabel("Patience factor δ_buyer")
    ax.set_ylabel("Deal rate")
    ax.legend(handles=color_handles, fontsize=9, title="Seller patience")
    ax.grid(alpha=0.3)

    style_handles_p2 = [
        mlines.Line2D([], [], color="gray", lw=2,   ls="-",  label="MADDPG learned"),
        mlines.Line2D([], [], color="gray", lw=1.5, ls="--", label="Rubinstein eq."),
        mlines.Line2D([], [], color="gray", lw=1.5, ls=":",  label="Nash bargaining"),
    ]
    ax = axes[0, 1]
    ax.set_ylim(0, 1)
    ax.set_title("Deal Price vs Theoretical Benchmarks")
    ax.set_xlabel("Patience factor δ_buyer")
    ax.set_ylabel("Deal price  (0=seller cost, 1=buyer value)")
    leg1 = ax.legend(handles=color_handles,   fontsize=8, title="Seller patience",
                     loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles_p2, fontsize=8, title="Line type",
              loc="lower left")
    ax.grid(alpha=0.3)


    style_handles_p3 = [
        mlines.Line2D([], [], color="gray", lw=2,   ls="-",  label="Buyer reward"),
        mlines.Line2D([], [], color="gray", lw=2,   ls=":",  label="Seller reward"),
        mlines.Line2D([], [], color="gray", lw=1.5, ls="--", label="Social welfare (sum)"),
    ]
    ax = axes[1, 0]
    ax.set_title("Agent Rewards and Social Welfare vs δ_buyer")
    ax.set_xlabel("Patience factor δ_buyer")
    ax.set_ylabel("Discounted reward")
    leg1 = ax.legend(handles=color_handles,   fontsize=8, title="Seller patience",
                     loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles_p3, fontsize=8, title="Line type",
              loc="lower left")
    ax.grid(alpha=0.3)

  
    ax = axes[1, 1]
    ax.bar(range(len(bar_labels)), bar_errors, color=bar_colors,
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_title("|Learned Price - Rubinstein Price|")
    ax.set_xlabel("(δ_buyer, δ_seller) pair")
    ax.set_ylabel("Absolute deviation")
    ax.grid(axis="y", alpha=0.3)
    valid_errors = [e for e in bar_errors if not np.isnan(e)]
    ax.set_ylim(0, max(valid_errors) * 1.4 if valid_errors else 0.3)
    for i, e in enumerate(bar_errors):
        if not np.isnan(e):
            ax.text(i, e + 0.002, f"{e:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "patience_asymmetric_lines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Asymmetric line plot saved → {path}")




def main(skip_train: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, "patience_results.npy")

    if skip_train and os.path.exists(results_path):
        print(f"Loading saved results from {results_path} …")
        saved = np.load(results_path, allow_pickle=True).item()
        sym_results  = saved["symmetric"]
        asym_results = saved["asymmetric"]
    else:
        sym_results  = run_symmetric_sweep()
        asym_results = run_asymmetric_sweep()
        np.save(results_path,
                {"symmetric": sym_results, "asymmetric": asym_results},
                allow_pickle=True)
        print(f"\nResults saved → {results_path}")

 
    plot_symmetric(sym_results)
    plot_heatmaps(asym_results)
    plot_asymmetric_lines(asym_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patience discount factor sweep")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Load previously saved results and only regenerate plots",
    )
    args = parser.parse_args()
    main(skip_train=args.skip_train)

