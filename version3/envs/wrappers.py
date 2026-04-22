"""
envs/wrappers.py
================
Custom Gymnasium wrapper for MarketMakerEnv.

ClipRewardWrapper – clips reward to [-r_max, +r_max] and optionally
                    applies tanh squashing for smoother gradients.
"""

from __future__ import annotations
import logging
import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Stabilises training by preventing extreme reward values.

    Parameters
    ----------
    env    : gym.Env  The environment to wrap.
    r_max  : float    Hard clip limit (default 1.0). Used when clip=True.
    scale  : float    Tanh denominator (default 1.0). Used when squash=True.
    clip   : bool     Enable hard clipping (default True).
    squash : bool     Enable tanh squashing (default False).
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

    def reward(self, reward: float) -> float:
        r = float(reward)

        if self._clip:
            r = float(np.clip(r, -self.r_max, self.r_max))

        if self._squash:
            r = float(np.tanh(r / self.scale))

        logger.debug("ClipRewardWrapper | raw=%.4f -> shaped=%.4f", reward, r)
        return r
