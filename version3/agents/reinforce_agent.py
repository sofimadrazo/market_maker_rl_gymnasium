"""
agents/reinforce_agent.py
=========================
REINFORCE (Monte Carlo Policy Gradient) agent implemented with numpy only.

Architecture
------------
Gaussian MLP policy:
    Input (7) → Dense(32, tanh) → Dense(16, tanh) → Dense(2, sigmoid) = μ
    σ = exp(log_std)   (trainable vector, separate from the network)
    action ~ clip(N(μ, σ²), 0, 1)

Algorithm (per episode)
-----------------------
1. Roll out one episode collecting (obs_t, action_t, reward_t).
2. Compute discounted returns G_t = Σ_{k≥t} γ^(k-t) r_k.
3. Normalise G_t to reduce variance.
4. Accumulate ∇θ log π(a_t|s_t) · G_t via manual backpropagation.
5. Gradient ascent: θ ← θ + α · ∇θ.
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class REINFORCEAgent:
    """
    Parameters
    ----------
    obs_dim : int    Observation dimension (default 7).
    act_dim : int    Action dimension (default 2).
    hidden  : tuple  Hidden layer sizes (default (32, 16)).
    lr      : float  Learning rate for gradient ascent.
    gamma   : float  Discount factor for returns.
    seed    : int    Random seed.
    """

    def __init__(
        self,
        obs_dim: int   = 7,
        act_dim: int   = 2,
        hidden:  tuple = (32, 16),
        lr:      float = 3e-4,
        gamma:   float = 0.99,
        seed:    int   = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr      = lr
        self.gamma   = gamma
        self._rng    = np.random.default_rng(seed)

        # Build weight matrices with He initialisation
        dims = [obs_dim] + list(hidden) + [act_dim]
        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(self._rng.normal(0.0, scale, (dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1]))

        # Log std (trainable): initial std ≈ exp(-1) ≈ 0.37
        self.log_std = np.full(act_dim, -1.0)

        logger.info(
            "REINFORCEAgent | layers=%s lr=%.2e gamma=%.3f",
            dims, lr, gamma,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run forward pass through the MLP.

        Returns
        -------
        mu          : np.ndarray (act_dim,)  – action mean ∈ (0, 1)
        activations : list of layer outputs (including input), needed for backprop.
        """
        x = obs.flatten().astype(np.float64)
        activations = [x]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ W + b
            if i < len(self.weights) - 1:
                x = np.tanh(z)
            else:
                x = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))  # sigmoid
            activations.append(x)
        return x, activations  # x = mu ∈ (0, 1)^act_dim

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Sample an action from the Gaussian policy π(·|obs).

        Returns
        -------
        action   : np.ndarray (act_dim,) clipped to [0, 1]
        log_prob : float  log π(action | obs)
        """
        mu, _ = self._forward(obs)
        std    = np.exp(self.log_std)
        noise  = self._rng.standard_normal(self.act_dim)
        action = np.clip(mu + std * noise, 0.0, 1.0).astype(np.float32)

        # Gaussian log-probability (before clipping — standard approximation)
        log_prob = float(
            -0.5 * np.sum(((action - mu) / (std + 1e-8)) ** 2)
            - np.sum(np.log(std + 1e-8))
        )
        return action, log_prob

    # ------------------------------------------------------------------
    # Discounted returns
    # ------------------------------------------------------------------

    def _compute_returns(self, rewards: List[float]) -> np.ndarray:
        """Compute normalised discounted returns G_t."""
        T = len(rewards)
        G = np.zeros(T)
        cumulative = 0.0
        for t in reversed(range(T)):
            cumulative = rewards[t] + self.gamma * cumulative
            G[t] = cumulative
        # Normalise for variance reduction
        std = G.std()
        if std > 1e-8:
            G = (G - G.mean()) / std
        return G

    # ------------------------------------------------------------------
    # Parameter update (manual backpropagation)
    # ------------------------------------------------------------------

    def update(
        self,
        episode_obs:     List[np.ndarray],
        episode_actions: List[np.ndarray],
        episode_rewards: List[float],
    ) -> float:
        """
        Update policy parameters using one episode of experience.

        Returns the mean episode return (for logging).
        """
        returns = self._compute_returns(episode_rewards)
        mean_return = float(np.mean(returns))

        n_layers = len(self.weights)
        grad_W       = [np.zeros_like(W) for W in self.weights]
        grad_b       = [np.zeros_like(b) for b in self.biases]
        grad_log_std = np.zeros_like(self.log_std)

        std = np.exp(self.log_std)

        for obs, action, G_t in zip(episode_obs, episode_actions, returns):
            mu, activations = self._forward(obs)
            action = action.astype(np.float64)

            # ── Gradient of log π w.r.t. μ ──────────────────────────────
            # d/dμ  log N(a; μ, σ²) = (a - μ) / σ²
            d_log_mu = (action - mu) / (std ** 2 + 1e-8)        # (act_dim,)

            # ── Gradient of log π w.r.t. log_std ───────────────────────
            # d/d(log σ)  log N = (a-μ)²/σ² - 1
            grad_log_std += G_t * ((action - mu) ** 2 / (std ** 2 + 1e-8) - 1.0)

            # ── Backprop through output layer (sigmoid) ─────────────────
            # μ = sigmoid(z_last),  dμ/dz = μ*(1-μ)
            last_out = activations[-1]           # = mu
            d_z = d_log_mu * last_out * (1.0 - last_out)         # (act_dim,)

            grad_W[n_layers - 1] += G_t * np.outer(activations[-2], d_z)
            grad_b[n_layers - 1] += G_t * d_z

            # ── Backprop through hidden layers (tanh) ───────────────────
            d_h = d_z @ self.weights[n_layers - 1].T
            for i in range(n_layers - 2, -1, -1):
                h = activations[i + 1]              # tanh output of layer i
                d_z = d_h * (1.0 - h ** 2)          # tanh derivative
                grad_W[i] += G_t * np.outer(activations[i], d_z)
                grad_b[i] += G_t * d_z
                if i > 0:
                    d_h = d_z @ self.weights[i].T

        # ── Gradient ascent ─────────────────────────────────────────────
        for i in range(n_layers):
            self.weights[i] += self.lr * grad_W[i]
            self.biases[i]  += self.lr * grad_b[i]

        self.log_std += self.lr * grad_log_std
        # Keep std in a reasonable range [exp(-3), exp(0.5)] ≈ [0.05, 1.65]
        self.log_std = np.clip(self.log_std, -3.0, 0.5)

        logger.debug("REINFORCEAgent | mean_return=%.4f", mean_return)
        return mean_return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {f"W{i}": W for i, W in enumerate(self.weights)}
        data.update({f"b{i}": b for i, b in enumerate(self.biases)})
        data["log_std"] = self.log_std
        np.savez(path, **data)
        logger.info("REINFORCEAgent | saved to %s.npz", path)

    @classmethod
    def load(
        cls,
        path:    str,
        obs_dim: int   = 7,
        act_dim: int   = 2,
        hidden:  tuple = (32, 16),
        lr:      float = 3e-4,
        gamma:   float = 0.99,
    ) -> "REINFORCEAgent":
        agent = cls(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, lr=lr, gamma=gamma)
        data  = np.load(path if path.endswith(".npz") else path + ".npz")
        agent.weights  = [data[f"W{i}"] for i in range(len(agent.weights))]
        agent.biases   = [data[f"b{i}"] for i in range(len(agent.biases))]
        agent.log_std  = data["log_std"]
        logger.info("REINFORCEAgent | loaded from %s", path)
        return agent
