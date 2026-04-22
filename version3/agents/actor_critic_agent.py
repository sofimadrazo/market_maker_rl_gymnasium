"""
agents/actor_critic_agent.py
============================
Monte Carlo Actor-Critic agent implemented with numpy only.

Architecture
------------
Actor  (policy):  Input(7) → Dense(32, tanh) → Dense(16, tanh) → Dense(2, sigmoid) = μ
                  + log_std(2) trainable vector
                  action ~ clip(N(μ, σ²), 0, 1)

Critic (value fn): Input(7) → Dense(32, tanh) → Dense(16, tanh) → Dense(1, linear) = V(s)

Update rule (per episode)
-------------------------
1. Roll out episode: collect (obs_t, action_t, reward_t).
2. Compute discounted returns  G_t = Σ_{k≥t} γ^(k-t) r_k.
3. Compute value estimates     V_t = critic(obs_t)   for all t.
4. Compute advantages          A_t = G_t − V_t.
5. Actor  gradient ascent:  ∇θ_actor  log π(a_t|s_t) · A_t
6. Critic gradient descent: minimize  Σ_t (G_t − V_t)²
   ↔ gradient ascent on  (G_t − V_t) · ∂V_t/∂θ_critic

Key improvement over REINFORCE
-------------------------------
A_t = G_t − V_t removes the "baseline" (expected return) from the update signal.
The critic learns what a "normal" return looks like for each state, so the actor
gradient only reacts to how much *better or worse* the episode was than expected.
This dramatically reduces variance without introducing bias.
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ActorCriticAgent:
    """
    Parameters
    ----------
    obs_dim   : Observation dimension (default 7).
    act_dim   : Action dimension (default 2).
    hidden    : Hidden layer sizes shared by actor and critic (default (32, 16)).
    actor_lr  : Learning rate for the actor (policy) network.
    critic_lr : Learning rate for the critic (value) network.
    gamma     : Discount factor for returns.
    seed      : Random seed.
    """

    def __init__(
        self,
        obs_dim:   int   = 7,
        act_dim:   int   = 2,
        hidden:    tuple = (32, 16),
        actor_lr:  float = 1e-4,
        critic_lr: float = 5e-4,
        gamma:     float = 0.99,
        seed:      int   = 0,
    ):
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.actor_lr  = actor_lr
        self.critic_lr = critic_lr
        self.gamma     = gamma
        self._rng      = np.random.default_rng(seed)

        dims_actor  = [obs_dim] + list(hidden) + [act_dim]
        dims_critic = [obs_dim] + list(hidden) + [1]

        self.actor_W:  List[np.ndarray] = []
        self.actor_b:  List[np.ndarray] = []
        self.critic_W: List[np.ndarray] = []
        self.critic_b: List[np.ndarray] = []

        for dims, W_list, b_list in [
            (dims_actor,  self.actor_W,  self.actor_b),
            (dims_critic, self.critic_W, self.critic_b),
        ]:
            for i in range(len(dims) - 1):
                scale = np.sqrt(2.0 / dims[i])
                W_list.append(self._rng.normal(0.0, scale, (dims[i], dims[i + 1])))
                b_list.append(np.zeros(dims[i + 1]))

        # Trainable log-std for the Gaussian policy (actor)
        self.log_std = np.full(act_dim, -1.0)

        logger.info(
            "ActorCriticAgent | actor_lr=%.2e  critic_lr=%.2e  gamma=%.3f",
            actor_lr, critic_lr, gamma,
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _forward(
        self,
        obs:    np.ndarray,
        W_list: List[np.ndarray],
        b_list: List[np.ndarray],
        output: str = "sigmoid",
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generic forward pass used by both actor and critic.

        output : "sigmoid"  → last layer uses sigmoid  (actor, output ∈ (0,1))
                 "linear"   → last layer is identity   (critic, output ∈ ℝ)
        """
        x = obs.flatten().astype(np.float64)
        activations = [x]
        n = len(W_list)
        for i, (W, b) in enumerate(zip(W_list, b_list)):
            z = x @ W + b
            if i < n - 1:
                x = np.tanh(z)
            elif output == "sigmoid":
                x = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            else:
                x = z        # linear output for critic
            activations.append(x)
        return x, activations

    def _forward_actor(self, obs: np.ndarray):
        return self._forward(obs, self.actor_W, self.actor_b, output="sigmoid")

    def _forward_critic(self, obs: np.ndarray):
        return self._forward(obs, self.critic_W, self.critic_b, output="linear")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Sample action ~ π(·|obs). Returns (action, log_prob)."""
        mu, _ = self._forward_actor(obs)
        std    = np.exp(self.log_std)
        noise  = self._rng.standard_normal(self.act_dim)
        action = np.clip(mu + std * noise, 0.0, 1.0).astype(np.float32)
        log_prob = float(
            -0.5 * np.sum(((action - mu) / (std + 1e-8)) ** 2)
            - np.sum(np.log(std + 1e-8))
        )
        return action, log_prob

    # ------------------------------------------------------------------
    # Discounted returns
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: List[float]) -> np.ndarray:
        T = len(rewards)
        G = np.zeros(T)
        cumulative = 0.0
        for t in reversed(range(T)):
            cumulative = rewards[t] + self.gamma * cumulative
            G[t] = cumulative
        return G

    # ------------------------------------------------------------------
    # Generic backprop
    # ------------------------------------------------------------------

    def _backprop(
        self,
        activations: List[np.ndarray],
        d_out:       np.ndarray,
        W_list:      List[np.ndarray],
        output:      str,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute parameter gradients given the error signal at the output.

        d_out  : gradient at the output layer (before activation derivative).
        output : "sigmoid" or "linear" — determines last-layer derivative.
        """
        n = len(W_list)
        grad_W = [np.zeros_like(W) for W in W_list]
        grad_b = [np.zeros_like(b) for b in (self.actor_b if W_list is self.actor_W else self.critic_b)]

        last_out = activations[-1]
        if output == "sigmoid":
            d_z = d_out * last_out * (1.0 - last_out)
        else:
            d_z = d_out * np.ones_like(last_out)  # linear: derivative = 1

        grad_W[n - 1] = np.outer(activations[-2], d_z)
        grad_b[n - 1] = d_z

        d_h = d_z @ W_list[n - 1].T
        for i in range(n - 2, -1, -1):
            h = activations[i + 1]
            d_z = d_h * (1.0 - h ** 2)        # tanh derivative
            grad_W[i] = np.outer(activations[i], d_z)
            grad_b[i] = d_z
            if i > 0:
                d_h = d_z @ W_list[i].T

        return grad_W, grad_b

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_grad_norm(
        grads_W: List[np.ndarray],
        grads_b: List[np.ndarray],
        max_norm: float = 1.0,
    ) -> None:
        """Clip gradient list in-place to a global max L2 norm."""
        total_sq = sum(np.sum(g ** 2) for g in grads_W + grads_b)
        norm = float(np.sqrt(total_sq))
        if norm > max_norm:
            scale = max_norm / norm
            for g in grads_W + grads_b:
                g *= scale

    def update(
        self,
        episode_obs:     List[np.ndarray],
        episode_actions: List[np.ndarray],
        episode_rewards: List[float],
    ) -> Tuple[float, float]:
        """
        Update actor and critic from one full episode.

        Numerical stability measures applied here:
          1. Advantages are normalised (zero-mean, unit-std) per episode.
          2. Gradients are averaged over T steps (not summed).
          3. Gradient norms are clipped before each parameter update.

        Returns
        -------
        (mean_raw_advantage, mean_value_error)  – for logging.
        """
        returns = self._compute_returns(episode_rewards)
        T       = len(returns)
        std     = np.exp(self.log_std)

        # ── 1. Pre-compute all critic values and advantages ──────────────
        values = np.array([
            float(self._forward_critic(obs)[0][0]) for obs in episode_obs
        ])
        advantages_raw = returns - values

        # Normalise advantages (variance reduction + numerical stability)
        adv_std = advantages_raw.std()
        advantages = (advantages_raw - advantages_raw.mean()) / (adv_std + 1e-8)

        # ── 2. Accumulate gradients over the episode ─────────────────────
        n_actor  = len(self.actor_W)
        n_critic = len(self.critic_W)
        g_actor_W  = [np.zeros_like(W) for W in self.actor_W]
        g_actor_b  = [np.zeros_like(b) for b in self.actor_b]
        g_critic_W = [np.zeros_like(W) for W in self.critic_W]
        g_critic_b = [np.zeros_like(b) for b in self.critic_b]
        g_log_std  = np.zeros_like(self.log_std)

        for obs, action, A_norm, G_t, V_t in zip(
            episode_obs, episode_actions, advantages, returns, values
        ):
            action = action.astype(np.float64)

            # Actor gradient: ∇ log π(a|s) · A_norm
            mu, actor_act = self._forward_actor(obs)
            d_log_mu   = (action - mu) / (std ** 2 + 1e-8)
            g_log_std += A_norm * ((action - mu) ** 2 / (std ** 2 + 1e-8) - 1.0)

            gW, gb = self._backprop(actor_act, d_log_mu * A_norm, self.actor_W, "sigmoid")
            for i in range(n_actor):
                g_actor_W[i] += gW[i]
                g_actor_b[i] += gb[i]

            # Critic gradient: ascent on (G_t − V_t)
            _, crit_act = self._forward_critic(obs)
            d_critic    = np.array([G_t - V_t])
            gW, gb = self._backprop(crit_act, d_critic, self.critic_W, "linear")
            for i in range(n_critic):
                g_critic_W[i] += gW[i]
                g_critic_b[i] += gb[i]

        # ── 3. Average over T steps ──────────────────────────────────────
        for i in range(n_actor):
            g_actor_W[i] /= T
            g_actor_b[i] /= T
        g_log_std /= T
        for i in range(n_critic):
            g_critic_W[i] /= T
            g_critic_b[i] /= T

        # ── 4. Clip gradients ────────────────────────────────────────────
        self._clip_grad_norm(g_actor_W,  g_actor_b,  max_norm=1.0)
        self._clip_grad_norm(g_critic_W, g_critic_b, max_norm=1.0)

        # ── 5. Apply gradients ───────────────────────────────────────────
        for i in range(n_actor):
            self.actor_W[i] += self.actor_lr * g_actor_W[i]
            self.actor_b[i] += self.actor_lr * g_actor_b[i]
        self.log_std += self.actor_lr * g_log_std
        self.log_std  = np.clip(self.log_std, -3.0, 0.5)

        for i in range(n_critic):
            self.critic_W[i] += self.critic_lr * g_critic_W[i]
            self.critic_b[i] += self.critic_lr * g_critic_b[i]

        return float(np.mean(advantages_raw)), float(np.mean(np.abs(returns - values)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {f"aW{i}": W for i, W in enumerate(self.actor_W)}
        data.update({f"ab{i}": b for i, b in enumerate(self.actor_b)})
        data.update({f"cW{i}": W for i, W in enumerate(self.critic_W)})
        data.update({f"cb{i}": b for i, b in enumerate(self.critic_b)})
        data["log_std"] = self.log_std
        np.savez(path, **data)
        logger.info("ActorCriticAgent | saved to %s.npz", path)

    @classmethod
    def load(
        cls,
        path:      str,
        obs_dim:   int   = 7,
        act_dim:   int   = 2,
        hidden:    tuple = (32, 16),
        actor_lr:  float = 1e-4,
        critic_lr: float = 5e-4,
        gamma:     float = 0.99,
    ) -> "ActorCriticAgent":
        agent = cls(
            obs_dim=obs_dim, act_dim=act_dim, hidden=hidden,
            actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma,
        )
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        agent.actor_W  = [data[f"aW{i}"] for i in range(len(agent.actor_W))]
        agent.actor_b  = [data[f"ab{i}"] for i in range(len(agent.actor_b))]
        agent.critic_W = [data[f"cW{i}"] for i in range(len(agent.critic_W))]
        agent.critic_b = [data[f"cb{i}"] for i in range(len(agent.critic_b))]
        agent.log_std  = data["log_std"]
        logger.info("ActorCriticAgent | loaded from %s", path)
        return agent
