"""
agents/heuristic_agent.py
=========================
Rule-based Market Maker agent that combines Fuzzy Logic and the
BFS Inventory Planner to produce sensible quotes without any training.
"""

from __future__ import annotations
import logging

import numpy as np

from core.fuzzy_controller  import FuzzySpreadController
from core.inventory_planner import InventoryLiquidationPlanner

logger = logging.getLogger(__name__)


class HeuristicMarketMakerAgent:
    """
    Heuristic agent that serves as a strong baseline

    Decision logic
    --------------
    1. Compute spread_multiplier via FuzzySpreadController
       - Decides how wide the spread should be based on volatility and inventory
    2. Look up optimal skew direction via InventoryLiquidationPlanner
       - Decides whether we should bias quotes to buy or to sell
    3. Combine both to produce [bid_offset, ask_offset] ∈ [0, 1]
       - Final action used by the environment to place quotes

    Parameters
    ----------
    max_inventory
        Maximum inventory allowed by the environment
    skew_strength
        How strongly we bias quotes when inventory is unbalanced.
    """

    def __init__(self, max_inventory: int = 10, skew_strength: float = 0.15):
        self.max_inventory = max_inventory
        self.skew_strength = skew_strength

        self.fuzzy   = FuzzySpreadController()
        self.planner = InventoryLiquidationPlanner(max_inventory)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert an observation vector into a trading action

        Observation indices used
        ------------------------
        obs[3] : inventory_norm  (signed, in [-1, 1])
        obs[4] : volatility_norm (in [0, 1])

        Returns
        -------
        np.ndarray of shape (2,)
            [bid_offset, ask_offset], each in [0.05, 1.0]
        """
        inv_norm = float(obs[3])
        vol_norm = float(obs[4])

        inventory = int(round(inv_norm * self.max_inventory))

        spread_mult = self.fuzzy.compute_spread_multiplier(vol_norm, abs(inv_norm))
        base_offset = float(np.clip(0.30 * spread_mult, 0.05, 1.0))

        plan_action = self.planner.get_action(inventory)
        skew        = self.skew_strength * plan_action

        bid_offset = float(np.clip(base_offset - skew, 0.05, 1.0))
        ask_offset = float(np.clip(base_offset + skew, 0.05, 1.0))

        action = np.array([bid_offset, ask_offset], dtype=np.float32)

        logger.debug(
            "HeuristicAgent | inv=%+d vol=%.3f mult=%.2f skew=%+.2f "
            "bid_off=%.3f ask_off=%.3f",
            inventory, vol_norm, spread_mult, skew, bid_offset, ask_offset,
        )

        return action
