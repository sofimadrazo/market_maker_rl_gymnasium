"""
core/data_generator.py
======================
Synthetic price series generator using Geometric Brownian Motion (GBM).
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_prices(
    n_steps:       int   = 1000,
    initial_price: float = 100.0,
    sigma:         float = 0.002,
    seed:          int   = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic mid-price series via Geometric Brownian Motion.

    GBM formula:  S_t = S_{t-1} * exp(σ * Z_t),  Z_t ~ N(0, 1)

    Parameters
    ----------
    n_steps       : int    Number of time steps to generate.
    initial_price : float  Starting price (S_0).
    sigma         : float  Per-step volatility (annualised ÷ √252 for daily).
    seed          : int    Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        mid_price  : float  Simulated mid-price.
        volatility : float  Rolling 20-step realised volatility (std of returns).
    """
    rng      = np.random.default_rng(seed)
    returns  = rng.normal(0.0, sigma, n_steps)
    prices   = initial_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"mid_price": prices})
    df["volatility"] = (
        df["mid_price"]
        .pct_change()
        .rolling(20)
        .std()
        .fillna(sigma)
    )

    logger.info(
        "SyntheticData | steps=%d  μ_price=%.2f  σ_vol=%.5f",
        n_steps, prices.mean(), df["volatility"].mean(),
    )
    return df
