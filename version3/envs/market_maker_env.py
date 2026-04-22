"""
Market Maker Agent Simulation Environment — Version 3 (corrected).

Fixes applied vs version2:
  BUG1 – Inventory signs reversed on fills (bid→+1, ask→-1)
  BUG2 – obs[1]/obs[2] were constant; now track last bid/ask offsets
  BUG3 – Fill probability was shared; now computed independently per side
  BUG4 – Reward based on realized spread capture instead of noisy mark-to-market
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.fuzzy_controller import FuzzySpreadController
from core.data_generator   import generate_synthetic_prices

logger = logging.getLogger(__name__)


class MarketMakerEnv(gym.Env):
    """
    Gymnasium environment that simulates a Market Maker posting
    bid/ask quotes around the mid-price of an asset.

    Observation Space  Box(7,)  – normalised market features
    Action Space       Box(2,)  – [bid_offset, ask_offset] ∈ [0, 1]

    Reward (v3)
    -----------
    r_t = realized_spread_t  −  λ·|inventory_t|  −  0.001·inventory_t²

    Where realized_spread_t = bid_offset * bid_filled + ask_offset * ask_filled.
    This gives a clean, dense signal: every fill earns the half-spread captured.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        render_mode:       Optional[str] = None,
        initial_price:     float  = 100.0,
        max_spread:        float  = 0.10,
        base_spread:       float  = 0.02,
        fill_probability:  float  = 0.50,
        max_inventory:     int    = 10,
        inventory_penalty: float  = 0.05,
        episode_length:    int    = 500,
        data_path:         Optional[str] = None,
        n_synthetic_steps: int    = 1000,
        seed:              int    = 42,
    ):
        super().__init__()

        self.render_mode       = render_mode
        self.initial_price     = initial_price
        self.max_spread        = max_spread
        self.base_spread       = base_spread
        self.fill_probability  = fill_probability
        self.max_inventory     = max_inventory
        self.inventory_penalty = inventory_penalty
        self.episode_length    = episode_length
        self.data_path         = data_path
        self.n_synthetic_steps = n_synthetic_steps
        self._seed             = seed

        self.fuzzy = FuzzySpreadController()

        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2,  dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low  = np.array([-1, 0, 0, -1, 0, -10, 0], dtype=np.float32),
            high = np.array([ 1, 1, 1,  1, 1,  10, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self._price_data      = None
        self._step_idx        = 0
        self._inventory       = 0
        self._cash            = 0.0
        self._episode_pnl     = 0.0
        self._start_idx       = 0
        self._vol_max         = 1.0
        # BUG2 FIX: track last action offsets for observation
        self._last_bid_norm   = 0.0
        self._last_ask_norm   = 0.0
        self._rng             = np.random.default_rng(seed)

        logger.info("MarketMakerEnv v3 initialised | max_inv=%d ep_len=%d",
                    max_inventory, episode_length)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._price_data = self._load_data()
        self._vol_max    = float(self._price_data["volatility"].max()) + 1e-9

        max_start = len(self._price_data) - self.episode_length - 1
        self._start_idx = int(self._rng.integers(0, max(1, max_start)))

        self._step_idx      = 0
        self._inventory     = 0
        self._cash          = 0.0
        self._episode_pnl   = 0.0
        # BUG2 FIX: reset last action state
        self._last_bid_norm = 0.0
        self._last_ask_norm = 0.0

        logger.info("ENV RESET | start_idx=%d mid=%.4f",
                    self._start_idx, self._mid())
        return self._obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        mid      = self._mid()
        vol_norm = self._current_vol() / self._vol_max
        inv_norm = abs(self._inventory) / self.max_inventory

        # Fuzzy spread adjustment
        spread_mult = self.fuzzy.compute_spread_multiplier(vol_norm, inv_norm)
        bid_offset  = float(action[0]) * self.max_spread * spread_mult
        ask_offset  = float(action[1]) * self.max_spread * spread_mult
        bid_price   = mid - bid_offset
        ask_price   = mid + ask_offset

        # BUG2 FIX: store normalised offsets for next observation
        self._last_bid_norm = float(np.clip(bid_offset / (self.max_spread + 1e-9), 0.0, 1.0))
        self._last_ask_norm = float(np.clip(ask_offset / (self.max_spread + 1e-9), 0.0, 1.0))

        # BUG3 FIX: independent fill probability per side
        # Tighter quote (smaller offset) → higher depth → higher fill prob
        bid_depth     = 1.0 - bid_offset / (self.max_spread * spread_mult + 1e-9)
        ask_depth     = 1.0 - ask_offset / (self.max_spread * spread_mult + 1e-9)
        bid_fill_prob = self.fill_probability * max(0.1, bid_depth)
        ask_fill_prob = self.fill_probability * max(0.1, ask_depth)
        bid_filled    = bool(self._rng.random() < bid_fill_prob)
        ask_filled    = bool(self._rng.random() < ask_fill_prob)

        # BUG1 FIX: correct inventory direction
        # bid_filled → MM bought (customer sold to MM): inventory UP, cash DOWN
        if bid_filled and self._inventory < self.max_inventory:
            self._inventory += 1
            self._cash      -= bid_price

        # ask_filled → MM sold (customer bought from MM): inventory DOWN, cash UP
        if ask_filled and self._inventory > -self.max_inventory:
            self._inventory -= 1
            self._cash      += ask_price

        # PnL (mark-to-market)
        total_pnl         = self._cash + self._inventory * mid
        pnl_step          = total_pnl - self._episode_pnl
        self._episode_pnl = total_pnl

        # BUG4 FIX: realized spread reward
        reward = self._reward(bid_filled, ask_filled, bid_offset, ask_offset)

        self._step_idx += 1
        terminated = self._step_idx >= self.episode_length

        info = self._info(
            bid_price=bid_price, ask_price=ask_price,
            bid_filled=bid_filled, ask_filled=ask_filled,
            pnl_step=pnl_step, spread_mult=spread_mult,
        )

        logger.debug(
            "STEP %04d | mid=%.4f bid=%.4f ask=%.4f inv=%+d pnl=%.4f r=%.4f",
            self._step_idx, mid, bid_price, ask_price,
            self._inventory, self._episode_pnl, reward,
        )

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, False, info

    def render(self) -> Optional[str]:
        line = (
            f"┌──────────────────────────────────────────────┐\n"
            f"│  Step: {self._step_idx:>4d}/{self.episode_length:<4d}  "
            f"Mid: {self._mid():>8.4f}              │\n"
            f"│  Inventory: {self._inventory:>+4d}   "
            f"PnL: {self._episode_pnl:>+10.4f}          │\n"
            f"└──────────────────────────────────────────────┘"
        )
        if self.render_mode == "human":
            print(line)
        return line

    def close(self) -> None:
        logger.info("MarketMakerEnv closed")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self):
        if self.data_path is not None:
            try:
                import pandas as pd
                df = pd.read_csv(self.data_path)
                df["volatility"] = (
                    df["mid_price"].pct_change().rolling(20).std().fillna(0.002)
                )
                logger.info("Data loaded from %s (%d rows)", self.data_path, len(df))
                return df
            except Exception as exc:
                logger.warning("Could not load %s (%s); using synthetic.", self.data_path, exc)
        return generate_synthetic_prices(
            n_steps=self.n_synthetic_steps,
            initial_price=self.initial_price,
            seed=int(self._rng.integers(0, 2**31)),
        )

    def _global_idx(self) -> int:
        return min(self._start_idx + self._step_idx, len(self._price_data) - 1)

    def _mid(self) -> float:
        return float(self._price_data["mid_price"].iloc[self._global_idx()])

    def _current_vol(self) -> float:
        return float(self._price_data["volatility"].iloc[self._global_idx()])

    def _obs(self) -> np.ndarray:
        obs = np.array([
            self._mid() / self.initial_price - 1.0,
            self._last_bid_norm,            # BUG2 FIX: last bid offset (normalised)
            self._last_ask_norm,            # BUG2 FIX: last ask offset (normalised)
            self._inventory / self.max_inventory,
            self._current_vol() / self._vol_max,
            self._episode_pnl / self.initial_price,
            1.0 - self._step_idx / self.episode_length,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _info(self, **kwargs) -> Dict[str, Any]:
        base = {
            "step":        self._step_idx,
            "mid_price":   self._mid(),
            "inventory":   self._inventory,
            "episode_pnl": self._episode_pnl,
        }
        base.update(kwargs)
        return base

    def _reward(
        self,
        bid_filled: bool,
        ask_filled: bool,
        bid_offset: float,
        ask_offset: float,
    ) -> float:
        # BUG4 FIX: realized spread capture as primary signal
        realized = bid_offset * float(bid_filled) + ask_offset * float(ask_filled)
        inv_cost = (
            self.inventory_penalty * abs(self._inventory)
            + 0.001 * self._inventory ** 2
        )
        return float(realized - inv_cost)
