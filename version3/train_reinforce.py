"""
train_reinforce.py
==================
Training loop for the REINFORCE policy-gradient agent on MarketMakerEnv v3.

Usage
-----
    cd market_maker_rl_gymnasium/version3
    python train_reinforce.py

Output
------
    reinforce_weights.npz   – trained policy weights (loaded by main.py)
"""

from __future__ import annotations
import logging
import sys
import time

import numpy as np

from envs.market_maker_env  import MarketMakerEnv
from envs.wrappers          import ClipRewardWrapper
from agents.reinforce_agent import REINFORCEAgent

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EPISODES   = 5000     # total training episodes
EPISODE_LEN  = 300      # steps per episode
GAMMA        = 0.99     # discount factor
LR           = 1e-4     # learning rate
LOG_EVERY    = 100      # print progress every N episodes
WEIGHTS_PATH = "reinforce_weights"  # saved as .npz
CLIP_R_MAX   = 0.5      # ClipRewardWrapper hard limit


def run_episode(
    agent: REINFORCEAgent,
    env:   MarketMakerEnv,
    seed:  int,
    train: bool = True,
) -> tuple:
    """
    Execute one full episode.

    Returns
    -------
    (episode_obs, episode_actions, episode_rewards, total_reward, final_pnl)
    """
    obs, _ = env.reset(seed=seed)
    done   = False

    ep_obs, ep_actions, ep_rewards = [], [], []
    total_reward = 0.0

    while not done:
        if train:
            action, _ = agent.select_action(obs)
        else:
            # Deterministic evaluation: use mean of policy
            mu, _ = agent._forward(obs)
            action = mu.astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_obs.append(obs.copy())
        ep_actions.append(action.copy())
        ep_rewards.append(reward)
        total_reward += reward

    return ep_obs, ep_actions, ep_rewards, total_reward, info["episode_pnl"]


def main() -> None:
    print("=" * 60)
    print("  REINFORCE Training — MarketMakerEnv v3")
    print("=" * 60)

    raw_env = MarketMakerEnv(episode_length=EPISODE_LEN)
    env     = ClipRewardWrapper(raw_env, r_max=CLIP_R_MAX)
    agent   = REINFORCEAgent(lr=LR, gamma=GAMMA, seed=42)

    print(f"  ClipRewardWrapper active | r_max={CLIP_R_MAX}")

    reward_history: list = []
    pnl_history:    list = []

    t0 = time.time()

    for ep in range(1, N_EPISODES + 1):
        ep_obs, ep_actions, ep_rewards, total_reward, final_pnl = run_episode(
            agent, env, seed=ep, train=True
        )

        # Update policy with this episode's experience
        agent.update(ep_obs, ep_actions, ep_rewards)

        reward_history.append(total_reward)
        pnl_history.append(final_pnl)

        if ep % LOG_EVERY == 0:
            recent_reward = np.mean(reward_history[-LOG_EVERY:])
            recent_pnl    = np.mean(pnl_history[-LOG_EVERY:])
            elapsed       = time.time() - t0
            print(
                f"  Ep {ep:>5d}/{N_EPISODES} | "
                f"Avg reward (last {LOG_EVERY}): {recent_reward:>+8.4f} | "
                f"Avg PnL: {recent_pnl:>+8.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            sys.stdout.flush()

    env.close()

    # Save trained weights
    agent.save(WEIGHTS_PATH)
    print(f"\n[OK] Training complete. Weights saved -> {WEIGHTS_PATH}.npz")
    print(f"   Final avg reward (last 200 eps): {np.mean(reward_history[-200:]):>+.4f}")
    print(f"   Final avg PnL    (last 200 eps): {np.mean(pnl_history[-200:]):>+.4f}")


if __name__ == "__main__":
    main()
