"""
version2/example_wrappers.py
============================
Demo script for version2. Runs the same 3-agent comparison as version1/main.py
but inside a ClipRewardWrapper environment.

Shows side by side:
  - Version 1 behaviour : raw rewards (no wrapper)
  - Version 2 behaviour : rewards clipped by ClipRewardWrapper

Agents compared
---------------
  1. Random        – samples uniformly from the action space.
  2. TightSpread   – always posts minimum-width quotes.
  3. Heuristic     – fuzzy + BFS-guided quotes.

Run from version2/:
    cd version2
    python3 example_wrappers.py
"""

from __future__ import annotations
import logging
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

from envs.market_maker_env  import MarketMakerEnv
from envs.wrappers          import ClipRewardWrapper
from agents.heuristic_agent import HeuristicMarketMakerAgent
from utils.hyperparam_grid  import generate_hyperparam_grid


# ===========================================================================
# Policy definitions  (identical to version1/main.py)
# ===========================================================================

def random_policy(obs: np.ndarray, env) -> np.ndarray:
    """Uniformly random bid/ask offsets."""
    return env.action_space.sample()


def tight_spread_policy(obs: np.ndarray, env) -> np.ndarray:
    """Always post minimum-width quotes."""
    return np.array([0.05, 0.05], dtype=np.float32)


def make_heuristic_policy(max_inventory: int = 10):
    """Factory that returns a callable wrapping HeuristicMarketMakerAgent."""
    agent = HeuristicMarketMakerAgent(max_inventory=max_inventory)
    def policy(obs: np.ndarray, env) -> np.ndarray:
        return agent.select_action(obs)
    return policy


# ===========================================================================
# Evaluation harness
# ===========================================================================

def evaluate_policy(
    policy_fn,
    n_episodes:  int   = 5,
    episode_len: int   = 300,
    r_max:       float = 1.0,
    use_wrapper: bool  = True,
    seed_offset: int   = 0,
) -> dict:
    """
    Run ``n_episodes`` with ``policy_fn``.

    Parameters
    ----------
    use_wrapper : bool
        True  → wrap env with ClipRewardWrapper (version2 behaviour).
        False → bare env (version1 behaviour for comparison).
    """
    base_env = MarketMakerEnv(episode_length=episode_len)
    env = ClipRewardWrapper(base_env, r_max=r_max) if use_wrapper else base_env

    rewards, pnls, inventories, pnl_curves = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + seed_offset)
        total_reward = 0.0
        curve = []
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


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    N_EP      = 5
    EP_LEN    = 300
    R_MAX     = 1.0

    policies = {
        "Random":      random_policy,
        "TightSpread": tight_spread_policy,
        "Heuristic":   make_heuristic_policy(max_inventory=10),
    }

    # Run both versions for comparison
    results_v1 = {}   # bare env (no wrapper)
    results_v2 = {}   # ClipRewardWrapper

    for name, fn in policies.items():
        print(f"\n▶  Evaluating: {name} ...")
        t0 = time.time()
        results_v1[name] = evaluate_policy(fn, N_EP, EP_LEN, use_wrapper=False)
        results_v2[name] = evaluate_policy(fn, N_EP, EP_LEN, r_max=R_MAX, use_wrapper=True)
        print(f"   Done in {time.time() - t0:.2f}s")

    # ── Summary table ────────────────────────────────────────────────────
    for label, results in [("Version 1 — no wrapper", results_v1),
                            ("Version 2 — ClipRewardWrapper", results_v2)]:
        print(f"\n{'=' * 64}")
        print(f"  {label}")
        print(f"{'=' * 64}")
        print(f"  {'Policy':<14} {'Mean Reward':>12} {'Mean PnL':>10} {'Mean |Inv|':>12}")
        print(f"  {'-' * 52}")
        for name, res in results.items():
            print(
                f"  {name:<14} "
                f"{res['rewards'].mean():>+12.4f} "
                f"{res['pnls'].mean():>+10.4f} "
                f"{res['inventories'].mean():>12.2f}"
            )

    # ── PnL curves plot ───────────────────────────────────────────────────
    colours = {"Random": "#e74c3c", "TightSpread": "#3498db", "Heuristic": "#2ecc71"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

    for col, (name, _) in enumerate(policies.items()):
        for row, (results, row_label) in enumerate([
            (results_v1, "v1 — no wrapper"),
            (results_v2, f"v2 — ClipReward (r_max={R_MAX})"),
        ]):
            ax = axes[row][col]
            res = results[name]
            for i, curve in enumerate(res["pnl_curves"]):
                ax.plot(curve, alpha=0.5, color=colours[name],
                        label=f"Ep {i+1}" if i < 3 else None)
            valid = [c for c in res["pnl_curves"] if len(c) == EP_LEN]
            if valid:
                ax.plot(np.mean(valid, axis=0), lw=2.5,
                        color=colours[name], linestyle="--", label="Mean")
            ax.axhline(0, color="black", lw=0.8, linestyle=":")
            ax.set_title(f"{name} — {row_label}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative PnL")
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)

    fig.suptitle("Version 1 vs Version 2 (ClipRewardWrapper) — Cumulative PnL",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("policy_comparison_v2.png", dpi=150, bbox_inches="tight")
    print("\n✔  Plot saved → policy_comparison_v2.png")

    # ── Hyperparameter grid preview ───────────────────────────────────────
    grid = generate_hyperparam_grid()
    print(f"\n── Hyperparam grid: {len(grid)} combinations (first 5 shown) ──")
    for cfg in grid[:5]:
        print(f"   {cfg}")

    print("\n✔  Done.\n")
