"""
agents/dqn_agent.py
===================
Deep Q-Network (DQN) agent implemented with numpy only.

Key components
--------------
Q-network      : MLP  obs(7) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(N_ACTIONS, linear)
                 Maps each observation to a Q-value per discrete action.

Target network : Identical architecture, weights frozen and synced every
                 TARGET_UPDATE episodes. Prevents the moving-target instability.

Replay buffer  : Stores (s, a, r, s', done) transitions. Mini-batch sampling
                 breaks temporal correlations that destabilise online training.

Epsilon-greedy : Starts fully random (ε=1.0), decays exponentially to ε_min.
                 Exploration → exploitation transition.

Discrete action space
---------------------
The environment action space is Box(2,) ∈ [0,1]².
DQN requires discrete actions, so we define a 5×5 grid:
    bid_values × ask_values = 25 combinations.

Bellman update (per mini-batch)
--------------------------------
    y_i = r_i + γ · max_a' Q_target(s'_i, a')  (if not terminal)
    y_i = r_i                                    (if terminal)
    Loss = mean( (y_i − Q(s_i, a_i))² )
    Gradient ascent on −Loss w.r.t. θ_Q
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# ── Discrete action space ──────────────────────────────────────────────────────
_BID_VALUES  = [0.10, 0.25, 0.50, 0.75, 0.90]
_ASK_VALUES  = [0.10, 0.25, 0.50, 0.75, 0.90]
DISCRETE_ACTIONS: List[Tuple[float, float]] = [
    (b, a) for b in _BID_VALUES for a in _ASK_VALUES
]  # 25 actions


# ── Replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer storing (obs, action_idx, reward, next_obs, done)."""

    def __init__(self, capacity: int = 10_000):
        self.buffer:   deque = deque(maxlen=capacity)
        self._rng = np.random.default_rng(0)

    def push(
        self,
        obs:        np.ndarray,
        action_idx: int,
        reward:     float,
        next_obs:   np.ndarray,
        done:       bool,
    ) -> None:
        self.buffer.append((obs.copy(), action_idx, reward, next_obs.copy(), done))

    def sample(self, batch_size: int):
        idxs  = self._rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float64),
            np.array(acts,     dtype=np.int32),
            np.array(rews,     dtype=np.float64),
            np.array(next_obs, dtype=np.float64),
            np.array(dones,    dtype=np.float64),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Parameters
    ----------
    obs_dim       : Observation dimension.
    n_actions     : Number of discrete actions (default 25).
    hidden        : Hidden layer sizes (default (64, 64)).
    lr            : Q-network learning rate.
    gamma         : Discount factor.
    eps_start     : Initial epsilon for ε-greedy exploration.
    eps_end       : Minimum epsilon.
    eps_decay     : Multiplicative decay applied after each episode.
    buffer_size   : Replay buffer capacity.
    batch_size    : Mini-batch size for each gradient update.
    target_update : Sync target network every N episodes.
    seed          : Random seed.
    """

    def __init__(
        self,
        obs_dim:       int   = 7,
        n_actions:     int   = 25,
        hidden:        tuple = (64, 64),
        lr:            float = 5e-4,
        gamma:         float = 0.99,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     float = 0.997,
        buffer_size:   int   = 10_000,
        batch_size:    int   = 64,
        target_update: int   = 10,
        seed:          int   = 0,
    ):
        self.obs_dim       = obs_dim
        self.n_actions     = n_actions
        self.lr            = lr
        self.gamma         = gamma
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self._rng          = np.random.default_rng(seed)
        self.buffer        = ReplayBuffer(buffer_size)
        self._episodes     = 0

        dims = [obs_dim] + list(hidden) + [n_actions]

        # Q-network and target network (identical architecture)
        self.q_W:  List[np.ndarray] = []
        self.q_b:  List[np.ndarray] = []
        self.tq_W: List[np.ndarray] = []
        self.tq_b: List[np.ndarray] = []

        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])   # He initialisation for ReLU
            W = self._rng.normal(0.0, scale, (dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.q_W.append(W.copy());  self.q_b.append(b.copy())
            self.tq_W.append(W.copy()); self.tq_b.append(b.copy())

        logger.info(
            "DQNAgent | arch=%s  lr=%.2e  eps_decay=%.4f  target_update=%d",
            dims, lr, eps_decay, target_update,
        )

    # ------------------------------------------------------------------
    # Forward pass (ReLU hidden, linear output)
    # ------------------------------------------------------------------

    def _forward(
        self,
        obs:    np.ndarray,
        W_list: List[np.ndarray],
        b_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Returns (q_values, activations_list) for backprop."""
        x = obs.reshape(-1).astype(np.float64)
        activations = [x]
        n = len(W_list)
        for i, (W, b) in enumerate(zip(W_list, b_list)):
            z = x @ W + b
            x = np.maximum(0.0, z) if i < n - 1 else z   # ReLU or linear
            activations.append(x)
        return x, activations

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        q, _ = self._forward(obs, self.q_W, self.q_b)
        return q

    def target_q_values(self, obs: np.ndarray) -> np.ndarray:
        q, _ = self._forward(obs, self.tq_W, self.tq_b)
        return q

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection. Returns action index."""
        if not greedy and self._rng.random() < self.eps:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    @staticmethod
    def index_to_action(idx: int) -> np.ndarray:
        """Convert discrete action index → continuous [bid_offset, ask_offset]."""
        b, a = DISCRETE_ACTIONS[idx]
        return np.array([b, a], dtype=np.float32)

    # ------------------------------------------------------------------
    # Learning step (mini-batch gradient descent)
    # ------------------------------------------------------------------

    def _learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)
        B = self.batch_size
        n_layers = len(self.q_W)

        # ── Compute TD targets ────────────────────────────────────────
        q_next = np.array([
            self.target_q_values(nobs_b[i]) for i in range(B)
        ])                                          # (B, n_actions)
        targets = rew_b + self.gamma * q_next.max(axis=1) * (1.0 - done_b)

        # ── Forward pass on current Q-network ────────────────────────
        q_all_list = []
        acts_all   = []
        for i in range(B):
            q, acts = self._forward(obs_b[i], self.q_W, self.q_b)
            q_all_list.append(q)
            acts_all.append(acts)

        q_all = np.array(q_all_list)                  # (B, n_actions)
        td_errors = targets - q_all[np.arange(B), act_b]  # (B,)
        loss = float(np.mean(td_errors ** 2))

        # ── Accumulate gradients (only for taken action per sample) ──
        grad_W = [np.zeros_like(W) for W in self.q_W]
        grad_b = [np.zeros_like(b) for b in self.q_b]

        for i in range(B):
            acts    = acts_all[i]
            d_out   = np.zeros(self.n_actions)
            d_out[act_b[i]] = -td_errors[i]          # dMSE/dQ = −δ

            # Backprop through linear output layer
            d_z = d_out                               # linear: derivative = 1
            grad_W[n_layers - 1] += np.outer(acts[-2], d_z)
            grad_b[n_layers - 1] += d_z

            # Backprop through ReLU hidden layers
            d_h = d_z @ self.q_W[n_layers - 1].T
            for j in range(n_layers - 2, -1, -1):
                h   = acts[j + 1]
                d_z = d_h * (h > 0).astype(np.float64)   # ReLU derivative
                grad_W[j] += np.outer(acts[j], d_z)
                grad_b[j] += d_z
                if j > 0:
                    d_h = d_z @ self.q_W[j].T

        # Average over batch + gradient descent (minimise loss → negate grad)
        for j in range(n_layers):
            self.q_W[j] -= self.lr * grad_W[j] / B
            self.q_b[j] -= self.lr * grad_b[j] / B

        return loss

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self) -> None:
        """Call once per episode: decay ε, sync target network."""
        self._episodes += 1
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        if self._episodes % self.target_update == 0:
            for j in range(len(self.q_W)):
                self.tq_W[j] = self.q_W[j].copy()
                self.tq_b[j] = self.q_b[j].copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {f"qW{i}": W for i, W in enumerate(self.q_W)}
        data.update({f"qb{i}": b for i, b in enumerate(self.q_b)})
        data["eps"] = np.array([self.eps])
        np.savez(path, **data)
        logger.info("DQNAgent | saved to %s.npz", path)

    @classmethod
    def load(
        cls,
        path:    str,
        **kwargs,
    ) -> "DQNAgent":
        agent = cls(**kwargs)
        data  = np.load(path if path.endswith(".npz") else path + ".npz")
        agent.q_W  = [data[f"qW{i}"] for i in range(len(agent.q_W))]
        agent.q_b  = [data[f"qb{i}"] for i in range(len(agent.q_b))]
        agent.tq_W = [w.copy() for w in agent.q_W]
        agent.tq_b = [b.copy() for b in agent.q_b]
        agent.eps  = float(data["eps"][0])
        logger.info("DQNAgent | loaded from %s", path)
        return agent
