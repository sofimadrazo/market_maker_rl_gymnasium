"""
train_actor_critic.py
=====================
Training loop for the Actor-Critic agent on MarketMakerEnv v3.

Usage
-----
    cd market_maker_rl_gymnasium/version3
    python train_actor_critic.py

Output
------
    actor_critic_weights.npz   -- loaded by main.py for evaluation
"""

from __future__ import annotations
import logging
import sys
import time

import numpy as np

from envs.market_maker_env     import MarketMakerEnv
from envs.wrappers             import ClipRewardWrapper
from agents.actor_critic_agent import ActorCriticAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_actor_critic")

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_EPISODES    = 1000
EPISODE_LEN   = 500
GAMMA         = 0.99
ACTOR_LR      = 1e-4
CRITIC_LR     = 5e-4   # critic learns faster: needs a good baseline quickly
LOG_EVERY     = 100
CLIP_R_MAX    = 0.5
WEIGHTS_PATH  = "actor_critic_weights"


def _policy_entropy(log_std: np.ndarray) -> float:
    """Differential entropy of a Gaussian policy: 0.5*k*(1+ln2π) + sum(log_std)."""
    k = len(log_std)
    return float(0.5 * k * (1.0 + np.log(2.0 * np.pi)) + np.sum(log_std))


def main() -> None:
    print("=" * 60)
    print("  Actor-Critic Training -- MarketMakerEnv v3")
    print("=" * 60)
    print(f"  actor_lr={ACTOR_LR}  critic_lr={CRITIC_LR}  episodes={N_EPISODES}")
    print(f"  ClipRewardWrapper active | r_max={CLIP_R_MAX}")

    logger.info(
        "Hyperparams | actor_lr=%.2e  critic_lr=%.2e  gamma=%.3f  "
        "ep_len=%d  n_eps=%d  clip_r_max=%.2f",
        ACTOR_LR, CRITIC_LR, GAMMA, EPISODE_LEN, N_EPISODES, CLIP_R_MAX,
    )

    raw_env = MarketMakerEnv(episode_length=EPISODE_LEN, inventory_penalty=0.3)
    env     = ClipRewardWrapper(raw_env, r_max=CLIP_R_MAX)
    agent   = ActorCriticAgent(
        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, seed=42
    )

    reward_history:    list = []
    pnl_history:       list = []
    advantage_history: list = []
    value_err_history: list = []

    # Extended metric buffers
    inv_history:       list = []   # mean |inventory| per episode
    fillrate_history:  list = []   # fraction of steps with >= 1 fill
    entropy_history:   list = []   # policy entropy
    bid_off_history:   list = []   # mean bid offset
    ask_off_history:   list = []   # mean ask offset
    realized_history:  list = []   # mean realized spread reward per step

    t0 = time.time()

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset(seed=ep)
        ep_obs, ep_actions, ep_rewards = [], [], []
        done = False
        total_reward = 0.0

        # Per-episode accumulators
        fill_count   = 0
        inv_sum      = 0.0
        bid_sum      = 0.0
        ask_sum      = 0.0
        realized_sum = 0.0
        step_count   = 0

        while not done:
            action, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(reward)
            total_reward += reward

            # Collect per-step metrics
            bid_filled = info.get("bid_filled", False)
            ask_filled = info.get("ask_filled", False)
            if bid_filled or ask_filled:
                fill_count += 1

            inv_sum  += abs(info.get("inventory", 0))
            bid_sum  += float(action[0])
            ask_sum  += float(action[1])

            # Realized reward = reward without the inventory penalty term
            # (approximation: track raw action values as proxy)
            realized_sum += reward + raw_env.inventory_penalty * abs(info.get("inventory", 0))
            step_count += 1

        mean_adv, mean_verr = agent.update(ep_obs, ep_actions, ep_rewards)

        reward_history.append(total_reward)
        pnl_history.append(info["episode_pnl"])
        advantage_history.append(mean_adv)
        value_err_history.append(mean_verr)

        # Extended metrics
        inv_history.append(inv_sum / max(step_count, 1))
        fillrate_history.append(100.0 * fill_count / max(step_count, 1))
        entropy_history.append(_policy_entropy(agent.log_std))
        bid_off_history.append(bid_sum / max(step_count, 1))
        ask_off_history.append(ask_sum / max(step_count, 1))
        realized_history.append(realized_sum / max(step_count, 1))

        if ep % LOG_EVERY == 0:
            r_avg      = np.mean(reward_history[-LOG_EVERY:])
            pnl_avg    = np.mean(pnl_history[-LOG_EVERY:])
            adv_avg    = np.mean(advantage_history[-LOG_EVERY:])
            verr       = np.mean(value_err_history[-LOG_EVERY:])
            inv_avg    = np.mean(inv_history[-LOG_EVERY:])
            fill_avg   = np.mean(fillrate_history[-LOG_EVERY:])
            ent        = np.mean(entropy_history[-LOG_EVERY:])
            bid_avg    = np.mean(bid_off_history[-LOG_EVERY:])
            ask_avg    = np.mean(ask_off_history[-LOG_EVERY:])

            elapsed = time.time() - t0

            # Console summary (two lines for readability)
            print(
                f"  Ep {ep:>5d}/{N_EPISODES} | "
                f"Reward: {r_avg:>+8.2f} | "
                f"PnL: {pnl_avg:>+7.2f} | "
                f"Adv: {adv_avg:>+7.2f} | "
                f"ValErr: {verr:>6.2f} | "
                f"{elapsed:.0f}s"
            )
            print(
                f"  {'':>14}"
                f"Inv: {inv_avg:>4.1f} | "
                f"Fill: {fill_avg:>4.1f}% | "
                f"BidOff: {bid_avg:.3f} | "
                f"AskOff: {ask_avg:.3f} | "
                f"Entropy: {ent:>+.3f}"
            )
            sys.stdout.flush()

            # Structured INFO log with all metrics
            logger.info(
                "Ep %d/%d | reward=%.4f pnl=%.4f adv=%.4f verr=%.4f "
                "inv=%.2f fill=%.1f%% entropy=%.4f "
                "bid_off=%.4f ask_off=%.4f eps=%ds",
                ep, N_EPISODES,
                r_avg, pnl_avg, adv_avg, verr,
                inv_avg, fill_avg, ent,
                bid_avg, ask_avg,
                int(elapsed),
            )

    env.close()
    agent.save(WEIGHTS_PATH)
    print(f"\n[OK] Training complete. Weights saved -> {WEIGHTS_PATH}.npz")
    print(f"     Final avg reward (last 200): {np.mean(reward_history[-200:]):>+.2f}")
    print(f"     Final avg PnL    (last 200): {np.mean(pnl_history[-200:]):>+.2f}")
    print(f"     Final avg inv    (last 200): {np.mean(inv_history[-200:]):.2f}")
    print(f"     Final fill rate  (last 200): {np.mean(fillrate_history[-200:]):.1f}%")
    print(f"     Final entropy    (last 200): {np.mean(entropy_history[-200:]):.4f}")

    logger.info(
        "Training finished | reward=%.4f pnl=%.4f inv=%.2f fill=%.1f%% entropy=%.4f",
        np.mean(reward_history[-200:]),
        np.mean(pnl_history[-200:]),
        np.mean(inv_history[-200:]),
        np.mean(fillrate_history[-200:]),
        np.mean(entropy_history[-200:]),
    )


if __name__ == "__main__":
    main()
