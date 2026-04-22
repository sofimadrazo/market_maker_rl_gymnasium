"""
envs/wrappers.py
================
Custom Gymnasium wrapper for MarketMakerEnv.

Wrapper implemented
-------------------
ClipRewardWrapper – clips reward to [-r_max, +r_max] and optionally
                    applies tanh squashing for smoother gradients.

Usage
-----
    from envs.market_maker_env import MarketMakerEnv
    from envs.wrappers import ClipRewardWrapper

    env = ClipRewardWrapper(MarketMakerEnv(), r_max=1.0)
"""

from __future__ import annotations
import logging
import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)


class ClipRewardWrapper(gym.RewardWrapper):
    """
    StabiliThis wrapper helps keep training stable by making sure the AI doesn't 
    get "distracted" by massive, outlier reward values.ses training by preventing extreme reward values.

    Two modes (can be combined)
    ---------------------------
    clip   :Caps the reward so it never goes above or below a set limit. [-r_max, +r_max].
    squash : Uses a tanh function to squeeze rewards into a nice (-1, +1) curve.

    Why this matters for Market Making
    -----------------------------------
    The raw reward (DeltaPnL - inventory_penalty + fuzzy_bonus) can spike
    sharply when a large fill occurs against a volatile price move.
    If we don't rein those spikes in, they can "shock" the neural 
    network and ruin the learning process.

    Parameters
    ----------
    env    : gym.Env  The environment to wrap.
    r_max  : float    The highest (and lowest) value allowed if clipping is on.->Hard clip limit (default 1.0). Used when clip=True.
    scale  : float    Controls how steep the tanh curve is when squashing ->Tanh denominator (default 1.0). Used when squash=True.
    clip   : bool     Turn hard clipping on or of (default True).
    squash : bool     Turn the tanh smoothing on or off (default False).

    Examples
    --------
    # Hard clip only
    env = ClipRewardWrapper(MarketMakerEnv(), r_max=1.0)

    # Tanh squash only (no hard clip)
    env = ClipRewardWrapper(MarketMakerEnv(), clip=False, squash=True, scale=0.5)

    # Both: clip first, then squash
    env = ClipRewardWrapper(MarketMakerEnv(), r_max=1.0, squash=True, scale=1.0)
    """

    def __init__(
        self,
        env:    gym.Env,
        r_max:  float = 1.0,
        scale:  float = 1.0,
        clip:   bool  = True,
        squash: bool  = False,
    ):
        super().__init__(env)

        assert r_max > 0, "r_max must be positive."
        assert scale > 0, "scale must be positive."

        self.r_max   = r_max
        self.scale   = scale
        self._clip   = clip
        self._squash = squash

        logger.info(
            "ClipRewardWrapper | clip=%s r_max=%.2f | squash=%s scale=%.2f",
            clip, r_max, squash, scale,
        )

    # ------------------------------------------------------------------
    def reward(self, reward: float) -> float:
        """
        Transform the raw reward from the environment. We take the "raw" reward 
        from the market and clean it up:

        Steps (in order)
        ----------------
        1. We chop off the extremes (if clipping is enabled) -> Hard clip to [-r_max, +r_max]  (if clip=True)
        2. We smooth the result through a tanh curve -> tanh squash: tanh(r / scale)   (if squash=True)
        """
        r = float(reward)

        if self._clip:
            r = float(np.clip(r, -self.r_max, self.r_max))

        if self._squash:
            r = float(np.tanh(r / self.scale))

        logger.debug("ClipRewardWrapper | raw=%.4f -> shaped=%.4f", reward, r)
        return r
