class InventoryLiquidationPlanner:
    """
    BFS-based planner that pre-computes the minimum number of steps required
    to reduce an inventory position to zero under discrete fill assumptions.

    The output is a lookup table:
        inventory_level -> optimal first action (-1, 0, +1)
    """

    def __init__(self, max_inventory: int, fill_size: int = 1):
        # Maximum inventory allowed by the environment
        # The planner will consider all states from -max_inventory to +max_inventory
        self.max_inventory = max_inventory
        # Each action changes inventory by this amount (usually 1 unit)
        self.fill_size     = fill_size
        # Lookup table: inventory -> best action to move toward 0
        self.plan: Dict[int, int] = {}
        # Build the plan once at initialization using BFS
        self._build_plan()

    def _build_plan(self) -> None:
        """
        Build the optimal plan using BFS starting from the goal state (inventory = 0).

        Idea:
        -----
        - Start from inventory = 0 (the goal).
        - Explore backwards: which inventory states can reach 0 in 1 step? in 2 steps? etc.
        - For each inventory level, store the *first* action that leads optimally toward 0.
        """
        goal  = 0
        # Queue holds pairs: (state, path_to_goal)
        # path_to_goal is the sequence of actions needed to reach 0
        queue = deque([(goal, [])])
        # 'visited' maps inventory levels to the action sequence that reaches 0
        visited: Dict[int, List[int]] = {goal: []}

        while queue:
            state, path = queue.popleft()
            # Try all possible actions: sell (-1), neutral (0), buy (+1)
            for action in (-1, 0, 1):
                # Reverse transition: from which previous state could we have come?
                prev_state = state - action * self.fill_size
                # Only consider valid inventory levels and avoid revisiting
                if (
                    -self.max_inventory <= prev_state <= self.max_inventory
                    and prev_state not in visited
                ):
                    # Prepend this action to the path
                    visited[prev_state] = [action] + path
                    # Add the new state to the BFS queue
                    queue.append((prev_state, visited[prev_state]))

        # Build the lookup table: for each inventory, store the first action of its path
        for inv, actions in visited.items():
            # If no actions (inv = 0), choose neutral (0)
            self.plan[inv] = actions[0] if actions else 0

        logger.info(
            "InventoryPlanner | plan built for inv in [%d, %d]",
            "InventoryPlanner | plan built for inv in [%d, %d]",
            -self.max_inventory, self.max_inventory,
        )

    def get_action(self, inventory: int) -> int:
        """
        Return the greedy action that moves the inventory toward 0.

        - If inventory > 0 → likely returns -1 (sell)
        - If inventory < 0 → likely returns +1 (buy)
        - If inventory = 0 → returns 0 (neutral)

        The value is clipped to valid inventory bounds.
        """
        return self.plan.get(int(np.clip(inventory,
                                         -self.max_inventory,
                                          self.max_inventory)), 0)
