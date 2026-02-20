"""
analysis_patience.py
────────────────────
Sweep buyer/seller patience discount factors and analyse the effect
on MADDPG-learned negotiation outcomes.

Two sub-analyses
────────────────
1. Symmetric sweep  – δ_b = δ_s = δ,  varied across SYMMETRIC_DELTAS
   Shows how joint patience shifts deal prices toward / away from 0.5.
   Key Rubinstein insight: as δ→1 (both infinitely patient), equilibrium
   price → 0.5 (Nash split).  As δ→0, seller extracts most surplus.

2. Asymmetric grid  – independent δ_b × δ_s grid
   Reveals bargaining-power asymmetry: the MORE patient agent captures
   more surplus regardless of role.  Rubinstein prediction:
       deal_price = (1 − δ_b) / (1 − δ_b·δ_s)
   is overlaid as a benchmark on each heat-map.

For each (δ_b, δ_s) configuration fresh MADDPG agents are trained from
scratch (fewer episodes than the baseline to keep the sweep tractable).

Usage
─────
    python analysis_patience.py              # run full sweep
    python analysis_patience.py --skip-train # load saved results & replot

Outputs (written to ./checkpoints/patience_analysis/)
─────────────────────────────────────────────────────
    patience_symmetric.png   – 4-panel line plots vs δ
    patience_heatmaps.png    – 4-panel heatmaps over δ_b × δ_s
    patience_results.npy     – raw result list (allow_pickle=True)
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from maddpg import MADDPGAgent
from negotiation_env import NegotiationEnv
from replay_buffer import ReplayBuffer

# ── Sweep config ───────────────────────────────────────────────────────────────

SYMMETRIC_DELTAS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
ASYM_DELTAS      = [0.50, 0.70, 0.85, 0.95]          # axis values for 2-D grid

# ── Training hyper-params (lighter than baseline for sweep tractability) ───────

N_EPISODES   = 300
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
NOISE_DECAY  = 1_500
N_EVAL       = 300          # deterministic episodes per config

BASE_ENV_KWARGS = dict(
    max_rounds  = 10,
    buyer_value = 1.0,
    seller_cost = 0.0,
    history_len = 3,
)

OUT_DIR = "checkpoints/patience_analysis"

# ── Core helpers (mirrored from train_baseline.py) ─────────────────────────────

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


# ── Main sweep function ────────────────────────────────────────────────────────

def train_and_evaluate(buyer_discount: float, seller_discount: float) -> dict:
    """
    Train fresh MADDPG agents for one (δ_b, δ_s) configuration then
    run N_EVAL deterministic episodes and return aggregate metrics.
    """
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

    # ── Evaluation ────────────────────────────────────────────────────────────
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


# ── Run sweeps ────────────────────────────────────────────────────────────────

def run_symmetric_sweep() -> list[dict]:
    """Vary δ_b = δ_s = δ across SYMMETRIC_DELTAS."""
    print("\n" + "=" * 60)
    print("  Symmetric sweep  (δ_b = δ_s = δ)")
    print("=" * 60)
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
    """Full δ_b × δ_s grid over ASYM_DELTAS."""
    print("\n" + "=" * 60)
    print("  Asymmetric sweep  (δ_b × δ_s grid)")
    print("=" * 60)
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


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_symmetric(results: list[dict]):
    """4-panel line plot: outcomes vs shared patience δ."""
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

    # ── Panel 1: Deal Closure Rate ────────────────────────────────────────────
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

    # ── Panel 2: Learned price vs Rubinstein & Nash ───────────────────────────
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

    # ── Panel 3: Per-agent rewards ────────────────────────────────────────────
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

    # ── Panel 4: Price deviation from Rubinstein ──────────────────────────────
    ax = axes[1, 1]
    ax.bar(
        [str(d) for d in deltas], errors,
        color=["#d73027" if e > 0.1 else "#fdae61" if e > 0.05 else "#1a9850"
               for e in errors],
        edgecolor="black", linewidth=0.6,
    )
    ax.set_title("|Learned Price − Rubinstein Price|")
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
    """4-panel heatmaps over the δ_b × δ_s grid."""
    n   = len(ASYM_DELTAS)
    idx = {d: i for i, d in enumerate(ASYM_DELTAS)}

    # Build 2-D arrays indexed [row=δ_s, col=δ_b]  (conventional: y=rows)
    deal_rate_grid   = np.full((n, n), np.nan)
    price_grid       = np.full((n, n), np.nan)
    welfare_grid     = np.full((n, n), np.nan)
    rub_grid         = np.full((n, n), np.nan)   # Rubinstein predicted price
    price_error_grid = np.full((n, n), np.nan)

    for r in results:
        i = idx[r["seller_discount"]]   # row
        j = idx[r["buyer_discount"]]    # col
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

    # ── Panel 1: Deal Closure Rate ────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(deal_rate_grid, vmin=0, vmax=1,
                   cmap="RdYlGn", origin="lower")
    _annotate(ax, deal_rate_grid)
    _setup_ax(ax, "Deal Closure Rate")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 2: Learned Deal Price ───────────────────────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(price_grid, vmin=0, vmax=1,
                   cmap="coolwarm", origin="lower")
    _annotate(ax, price_grid)
    _setup_ax(ax, "Learned Deal Price\n(blue=low/buyer-favored, red=high/seller-favored)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add Rubinstein contours on top
    try:
        cs = axes[0, 1].contour(
            rub_grid, levels=5, colors="black", linewidths=0.8, linestyles="--",
            origin="lower",
        )
        axes[0, 1].clabel(cs, inline=True, fontsize=7, fmt="Rub=%.2f")
    except Exception:
        pass

    # ── Panel 3: Social Welfare ───────────────────────────────────────────────
    ax = axes[1, 0]
    im = ax.imshow(welfare_grid, cmap="YlOrRd", origin="lower")
    _annotate(ax, welfare_grid)
    _setup_ax(ax, "Social Welfare (Buyer + Seller Reward)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Panel 4: Price deviation from Rubinstein ──────────────────────────────
    ax = axes[1, 1]
    im = ax.imshow(price_error_grid, vmin=0, cmap="Reds", origin="lower")
    _annotate(ax, price_error_grid)
    _setup_ax(ax, "|Learned Price − Rubinstein Price|")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "patience_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap plot saved  → {path}")


# ── Summary printout ──────────────────────────────────────────────────────────

def print_summary(sym_results: list[dict], asym_results: list[dict]):
    print("\n" + "=" * 70)
    print("  PATIENCE ANALYSIS – KEY FINDINGS")
    print("=" * 70)

    # Symmetric: which δ gives closest to Rubinstein?
    valid = [r for r in sym_results if not np.isnan(r["price_error"])]
    if valid:
        best = min(valid, key=lambda r: r["price_error"])
        worst = max(valid, key=lambda r: r["price_error"])
        print(f"\n  Symmetric sweep:")
        print(f"    Closest to Rubinstein eq.: δ={best['buyer_discount']:.2f}  "
              f"(error={best['price_error']:.4f})")
        print(f"    Furthest from Rubinstein:  δ={worst['buyer_discount']:.2f}  "
              f"(error={worst['price_error']:.4f})")

    # Asymmetric: patient buyer vs patient seller
    def _find(db, ds):
        for r in asym_results:
            if np.isclose(r["buyer_discount"], db) and np.isclose(r["seller_discount"], ds):
                return r
        return None

    print(f"\n  Asymmetric highlights (δ=0.95 patient  vs  δ=0.50 impatient):")
    case_a = _find(0.95, 0.50)   # patient buyer, impatient seller
    case_b = _find(0.50, 0.95)   # impatient buyer, patient seller
    if case_a:
        print(f"    Patient BUYER  (δ_b=0.95, δ_s=0.50): price={case_a['avg_deal_price']:.3f}  "
              f"Rub={case_a['rubinstein_price']:.3f}")
    if case_b:
        print(f"    Patient SELLER (δ_b=0.50, δ_s=0.95): price={case_b['avg_deal_price']:.3f}  "
              f"Rub={case_b['rubinstein_price']:.3f}")
    if case_a and case_b:
        print(f"    → Patient agent captures more surplus (lower price = buyer win,")
        print(f"      higher price = seller win)")

    print("\n  Rubinstein prediction: patient agent always gets better deal.")
    print("  Higher δ → more patient → willing to wait → opponent concedes more.")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

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

    print_summary(sym_results, asym_results)
    plot_symmetric(sym_results)
    plot_heatmaps(asym_results)
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patience discount factor sweep")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Load previously saved results and only regenerate plots",
    )
    args = parser.parse_args()
    main(skip_train=args.skip_train)
