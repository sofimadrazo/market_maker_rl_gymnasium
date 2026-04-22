from typing import Dict, List
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

class InventoryLiquidationPlanner:
    """
    BFS-based planner that pre-computes the minimum number of steps required
    to reduce an inventory position to zero under discrete fill assumptions.

    The output is a lookup table:
        inventory_level -> optimal first action (-1, 0, +1)
    """

    def __init__(self, max_inventory: int, fill_size: int = 1):
        self.max_inventory = max_inventory
        self.fill_size     = fill_size
        self.plan: Dict[int, int] = {}
        self._build_plan()

    def _build_plan(self) -> None:
        goal  = 0
        queue = deque([(goal, [])])
        visited: Dict[int, List[int]] = {goal: []}

        while queue:
            state, path = queue.popleft()
            for action in (-1, 0, 1):
                prev_state = state - action * self.fill_size
                if (
                    -self.max_inventory <= prev_state <= self.max_inventory
                    and prev_state not in visited
                ):
                    visited[prev_state] = [action] + path
                    queue.append((prev_state, visited[prev_state]))

        for inv, actions in visited.items():
            self.plan[inv] = actions[0] if actions else 0

        logger.info(
            "InventoryPlanner | plan built for inv in [%d, %d]",
            -self.max_inventory, self.max_inventory
        )

    def get_action(self, inventory: int) -> int:
        """
        Return the greedy action that moves the inventory toward 0.

        - If inventory > 0 → likely returns -1 (sell)
        - If inventory < 0 → likely returns +1 (buy)
        - If inventory = 0 → returns 0 (neutral)
        """
        return self.plan.get(int(np.clip(inventory,
                                         -self.max_inventory,
                                          self.max_inventory)), 0)
