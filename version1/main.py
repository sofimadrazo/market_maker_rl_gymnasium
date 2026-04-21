"""
main.py
=======
Main script that compares three different trading policies inside the
MarketMakerEnv environment

It runs several episodes for each policy, collects performance metrics,
plots the cumulative PnL curves, and prints a summary table

"""

from __future__ import annotations
import logging
import time

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend to save plots to file
import matplotlib.pyplot as plt
import numpy as np

from envs.market_maker_env  import MarketMakerEnv
from agents.heuristic_agent import HeuristicMarketMakerAgent
from utils.hyperparam_grid  import generate_hyperparam_grid

# Configure logging so the console output stays clean and readable
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Policy definitions
def random_policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
    """
    A completely random policy
    It simply samples a random action from the environment's action space
    """
    return env.action_space.sample()


def tight_spread_policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
    """
    A very simple deterministic policy
    It always posts the tightest possible bid/ask offsets
    """
    return np.array([0.05, 0.05], dtype=np.float32)


def make_heuristic_policy(max_inventory: int = 10):
    """
    Factory function that creates a HeuristicMarketMakerAgent and returns
    a callable policy function that wraps the agent's select_action method
    """
    agent = HeuristicMarketMakerAgent(max_inventory=max_inventory)

    def policy(obs: np.ndarray, env: MarketMakerEnv) -> np.ndarray:
        # Delegate the decision to the heuristic agent
        return agent.select_action(obs)

    return policy


# Evaluation harness

def evaluate_policy(
    policy_fn,
    n_episodes:  int = 5,
    episode_len: int = 300,
    seed_offset: int = 0,
) -> dict:
    """
    Runs several episodes using the provided policy function and collects
    performance metrics such as total reward, final PnL, inventory usage,
    and the full PnL curve for each episode

    Returns a dictionary with numpy arrays and lists containing all results
    """
    env = MarketMakerEnv(episode_length=episode_len)

    rewards, pnls, inventories, pnl_curves = [], [], [], []

    for ep in range(n_episodes):
        # Reset the environment with a different seed each episode
        obs, _ = env.reset(seed=ep + seed_offset)
        total_reward = 0.0
        curve: list = []
        done = False

        # Run one full episode
        while not done:
            action = policy_fn(obs, env)  # Get action from the policy
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            curve.append(info["episode_pnl"])
            done = terminated or truncated

        # Store episode-level metrics
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


# Main execution block

if __name__ == "__main__":
    # Number of episodes and length of each episode
    N_EP  = 5
    EP_LEN = 300

    # Dictionary of policies to evaluate
    policies = {
        "Random":      random_policy,
        "TightSpread": tight_spread_policy,
        "Heuristic":   make_heuristic_policy(max_inventory=10),
    }

    results: dict = {}

    # Evaluate each policy and measure execution time
    for name, fn in policies.items():
        print(f"\n▶  Evaluating policy: {name} ...")
        t0 = time.time()
        results[name] = evaluate_policy(fn, n_episodes=N_EP, episode_len=EP_LEN)
        print(f"   Done in {time.time() - t0:.2f}s")

    # Print a summary table with mean metrics for each policy
    print("\n" + "=" * 62)
    print(f"{'Policy':<14} {'Mean Reward':>12} {'Mean PnL':>10} {'Mean |Inv|':>12}")
    print("-" * 62)

    for name, res in results.items():
        print(
            f"{name:<14} "
            f"{res['rewards'].mean():>+12.4f} "
            f"{res['pnls'].mean():>+10.4f} "
            f"{res['inventories'].mean():>12.2f}"
        )

    print("=" * 62)

    # Plot cumulative PnL curves for each policy and save the figure
    colours = {"Random": "#e74c3c", "TightSpread": "#3498db", "Heuristic": "#2ecc71"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, (name, res) in zip(axes, results.items()):
        # Plot each episode's PnL curve
        for i, curve in enumerate(res["pnl_curves"]):
            ax.plot(curve, alpha=0.5, color=colours[name],
                    label=f"Ep {i+1}" if i < 3 else None)

        # Plot the mean curve if all episodes have the same length
        valid = [c for c in res["pnl_curves"] if len(c) == EP_LEN]
        if valid:
            ax.plot(np.mean(valid, axis=0), lw=2.5,
                    color=colours[name], linestyle="--", label="Mean")

        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.set_title(f"{name} Policy", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative PnL")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Market Maker – Cumulative PnL by Policy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✔  Plot saved → policy_comparison.png")

    # Show a preview of the hyperparameter grid
    grid = generate_hyperparam_grid()
    print(f"\n── Hyperparam grid: {len(grid)} combinations (first 5 shown) ──")
    for cfg in grid[:5]:
        print(f"   {cfg}")
