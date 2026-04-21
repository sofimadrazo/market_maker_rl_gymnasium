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
        # Store configuration parameters
        self.max_inventory = max_inventory
        self.skew_strength = skew_strength

        # Fuzzy controller that determines spread width based on volatility + inventory
        self.fuzzy = FuzzySpreadController()

        # Planner that decides whether we should buy, sell, or stay neutral
        self.planner = InventoryLiquidationPlanner(max_inventory)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert an observation vector into a trading action

        Observation indices used
        ------------------------
        obs[3] : inventory_norm  (signed, in [-1, 1])
                 - Our current inventory, normalized.
        obs[4] : volatility_norm (in [0, 1])
                 - Market volatility, normalized.

        Returns
        -------
        np.ndarray of shape (2,)
            [bid_offset, ask_offset], each in [0.05, 1.0]
            - These offsets determine how far our quotes are from mid-price.
        """

        # Extract normalized inventory and volatility from the observation
        inv_norm = float(obs[3])     # signed normalized inventory
        vol_norm = float(obs[4])     # normalized volatility

        # Convert normalized inventory back to actual inventory units
        inventory = int(round(inv_norm * self.max_inventory))

        # 1. FUZZY LOGIC: determine how wide the spread should be
        #    Higher volatility or higher inventory → wider spread

        spread_mult = self.fuzzy.compute_spread_multiplier(vol_norm, abs(inv_norm))

        # Base spread offset, clipped to stay within allowed range
        base_offset = float(np.clip(0.30 * spread_mult, 0.05, 1.0))


        # 2. INVENTORY PLANNER: decide skew direction
        #    +1: we want to buy (tighten bid, widen ask)
        #    -1: we want to sell (tighten ask, widen bid)
        #     0: neutral

        plan_action = self.planner.get_action(inventory)

        # Amount of skew applied to the base spread
        skew = self.skew_strength * plan_action

      
        # 3. Combine spread + skew to produce final bid/ask offsets
      
        bid_offset = float(np.clip(base_offset - skew, 0.05, 1.0))
        ask_offset = float(np.clip(base_offset + skew, 0.05, 1.0))

        # Final action returned to the environment
        action = np.array([bid_offset, ask_offset], dtype=np.float32)

        # Debug log for analysis and troubleshooting
        logger.debug(
            "HeuristicAgent | inv=%+d vol=%.3f mult=%.2f skew=%+.2f "
            "bid_off=%.3f ask_off=%.3f",
            inventory, vol_norm, spread_mult, skew, bid_offset, ask_offset,
        )

        return action
