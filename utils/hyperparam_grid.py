"""
utils/hyperparam_grid.py
========================
Cartesian hyperparameter grid generator using itertools.product.
Used for grid search and PSO initialisation (Entrega 2).

Author : Persona C
"""

from __future__ import annotations
import itertools
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_hyperparam_grid(
    base_spreads:   List[float] = (0.01, 0.02, 0.05),
    inv_penalties:  List[float] = (0.01, 0.10, 0.50),
    fill_probs:     List[float] = (0.30, 0.50, 0.70),
) -> List[Dict[str, float]]:
    """
    Generate all combinations of environment hyperparameters.

    Uses `itertools.product` to build the full Cartesian product of
    the three parameter axes.

    Parameters
    ----------
    base_spreads  : iterable  Values for `base_spread`.
    inv_penalties : iterable  Values for `inventory_penalty`.
    fill_probs    : iterable  Values for `fill_probability`.

    Returns
    -------
    List of dicts, each representing one hyperparameter configuration.

    Example
    -------
    >>> grid = generate_hyperparam_grid()
    >>> len(grid)
    27
    >>> grid[0]
    {'base_spread': 0.01, 'inventory_penalty': 0.01, 'fill_probability': 0.3}
    """
    keys   = ("base_spread", "inventory_penalty", "fill_probability")
    combos = list(itertools.product(base_spreads, inv_penalties, fill_probs))
    grid   = [dict(zip(keys, combo)) for combo in combos]

    logger.info("HyperparamGrid | %d combinations generated", len(grid))
    return grid