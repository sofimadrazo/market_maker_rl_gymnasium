"""
main.py — Version 3
====================
Compares four trading policies on the corrected MarketMakerEnv v3:
  1. Random        – uniform random actions
  2. TightSpread   – fixed tight quotes [0.05, 0.05]
  3. Heuristic     – Fuzzy + BFS rule-based agent
  4. Actor-Critic  – trained RL agent (requires actor_critic_weights.npz)

Run sequence
------------
    cd market_maker_rl_gymnasium/version3
    python train_actor_critic.py     # train the RL agent first
    python main.py                   # then compare all policies
"""

from __future__ import annotations
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.market_maker_env     import MarketMakerEnv
from envs.wrappers             import ClipRewardWrapper
from agents.heuristic_agent    import HeuristicMarketMakerAgent
from agents.actor_critic_agent import ActorCriticAgent
from utils.hyperparam_grid     import generate_hyperparam_grid
import train_actor_critic

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

AC_WEIGHTS_PATH = "actor_critic_weights.npz"

# ── Policy definitions ─────────────────────────────────────────────────────────

def random_policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
    return env.action_space.sample()


def tight_spread_policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
    return np.array([0.05, 0.05], dtype=np.float32)


def make_heuristic_policy(max_inventory: int = 10):
    agent = HeuristicMarketMakerAgent(max_inventory=max_inventory)

    def policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
        return agent.select_action(obs)

    return policy


def make_actor_critic_policy(weights_path: str):
    """Load trained Actor-Critic and return a deterministic evaluation policy."""
    agent = ActorCriticAgent.load(weights_path)

    def policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
        mu, _ = agent._forward_actor(obs)
        return mu.astype(np.float32)

    return policy


# ── Evaluation harness ─────────────────────────────────────────────────────────

EVAL_CLIP_R_MAX = 0.5   # same clip used during training for fair comparison


def evaluate_policy(
    policy_fn,
    n_episodes:  int = 5,
    episode_len: int = 300,
    seed_offset: int = 0,
) -> dict:
    env = ClipRewardWrapper(
        MarketMakerEnv(episode_length=episode_len, inventory_penalty=0.3),
        r_max=EVAL_CLIP_R_MAX,
    )

    rewards, pnls, inventories, pnl_curves = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)
        total_reward = 0.0
        curve: list  = []
        done = False

        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            curve.append(info["episode_pnl"])
            done = terminated or truncated

        rewards.append(total_reward)
        pnls.append(info["episode_pnl"])
        inventories.append(abs(info["inventory"]))
        pnl_curves.append(curve)

    env.close()

    return {
        "rewards":     np.array(rewards),
        "pnls":        np.array(pnls),
        "inventories": np.array(inventories),
        "pnl_curves":  pnl_curves,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_EP   = 5
    EP_LEN = 500

    # Build policy dict; add Actor-Critic only if weights exist
    policies: dict = {
        "Random":      random_policy,
        "TightSpread": tight_spread_policy,
        "Heuristic":   make_heuristic_policy(max_inventory=10),
    }

    if not os.path.exists(AC_WEIGHTS_PATH):
        print(f"[!] {AC_WEIGHTS_PATH} not found. Training Actor-Critic now...\n")
        train_actor_critic.main()

    policies["Actor-Critic"] = make_actor_critic_policy(AC_WEIGHTS_PATH)
    print(f"[OK] Actor-Critic weights loaded from {AC_WEIGHTS_PATH}")

    results: dict = {}

    for name, fn in policies.items():
        print(f"\n>> Evaluating: {name} ...")
        t0 = time.time()
        results[name] = evaluate_policy(fn, n_episodes=N_EP, episode_len=EP_LEN)
        print(f"   Done in {time.time() - t0:.2f}s")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print(f"{'Policy':<14} {'Mean Reward':>12} {'Mean PnL':>10} {'Mean |Inv|':>12}")
    print("-" * 66)

    for name, res in results.items():
        print(
            f"{name:<14} "
            f"{res['rewards'].mean():>+12.4f} "
            f"{res['pnls'].mean():>+10.4f} "
            f"{res['inventories'].mean():>12.2f}"
        )

    print("=" * 66)

    # ── Plot ───────────────────────────────────────────────────────────────────
    colour_map = {
        "Random":       "#e74c3c",
        "TightSpread":  "#3498db",
        "Heuristic":    "#2ecc71",
        "Actor-Critic": "#e67e22",
    }

    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        colour = colour_map.get(name, "#555555")
        for i, curve in enumerate(res["pnl_curves"]):
            ax.plot(curve, alpha=0.5, color=colour,
                    label=f"Ep {i+1}" if i < 3 else None)

        valid = [c for c in res["pnl_curves"] if len(c) == EP_LEN]
        if valid:
            ax.plot(np.mean(valid, axis=0), lw=2.5,
                    color=colour, linestyle="--", label="Mean")

        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative PnL")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "MarketMakerEnv v3 (bugs corregidos) — Cumulative PnL by Policy",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = "policy_comparison_v3.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Plot saved -> {out_path}")

    # ── Hyperparameter grid preview ────────────────────────────────────────────
    grid = generate_hyperparam_grid()
    print(f"\n-- Hyperparam grid: {len(grid)} combinations (first 5 shown) --")
    for cfg in grid[:5]:
        print(f"   {cfg}")
