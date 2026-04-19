"""
core/fuzzy_controller.py
========================
Mamdani Fuzzy Inference System that maps market conditions
(volatility, inventory imbalance) to a spread multiplier.

Author : Persona A
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class FuzzySpreadController:
    """
    Mamdani-style FIS with 9 rules.

    Inputs
    ------
    vol_norm : float  Normalised volatility  in [0, 1]
    inv_norm : float  Normalised |inventory| in [0, 1]

    Output
    ------
    spread_multiplier : float  in [~0.6, ~1.8]
        < 1.0  →  tighter than base spread
        = 1.0  →  base spread unchanged
        > 1.0  →  wider than base spread
    """

    # ------------------------------------------------------------------
    # Membership functions
    # ------------------------------------------------------------------

    @staticmethod
    def _trimf(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function."""
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return (x - a) / (b - a + 1e-12)
        return (c - x) / (c - b + 1e-12)

    @staticmethod
    def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function."""
        if x <= a or x >= d:
            return 0.0
        left  = min(1.0, (x - a) / (b - a + 1e-12))
        right = min(1.0, (d - x) / (d - c + 1e-12))
        return min(left, right)

    # ------------------------------------------------------------------
    # Input universes
    # ------------------------------------------------------------------

    def _volatility_mf(self, v: float) -> Dict[str, float]:
        """Returns membership degrees for each volatility linguistic label."""
        return {
            "LOW":    self._trapmf(v, 0.00, 0.00, 0.25, 0.50),
            "MEDIUM": self._trimf (v, 0.25, 0.50, 0.75),
            "HIGH":   self._trapmf(v, 0.50, 0.75, 1.00, 1.00),
        }

    def _inventory_mf(self, i: float) -> Dict[str, float]:
        """Returns membership degrees for each inventory linguistic label."""
        return {
            "BALANCED":  self._trapmf(i, 0.00, 0.00, 0.20, 0.40),
            "MODERATE":  self._trimf (i, 0.20, 0.50, 0.80),
            "EXTREME":   self._trapmf(i, 0.60, 0.80, 1.00, 1.00),
        }

    # ------------------------------------------------------------------
    # Rule base  (volatility, inventory) → output label
    # ------------------------------------------------------------------

    _RULES: Dict[Tuple[str, str], str] = {
        ("LOW",    "BALANCED"): "TIGHT",
        ("LOW",    "MODERATE"): "NORMAL",
        ("LOW",    "EXTREME"):  "WIDE",
        ("MEDIUM", "BALANCED"): "NORMAL",
        ("MEDIUM", "MODERATE"): "NORMAL",
        ("MEDIUM", "EXTREME"):  "WIDE",
        ("HIGH",   "BALANCED"): "WIDE",
        ("HIGH",   "MODERATE"): "WIDE",
        ("HIGH",   "EXTREME"):  "WIDE",
    }

    # Output singleton centroids for defuzzification
    _CENTROIDS: Dict[str, float] = {"TIGHT": 0.6, "NORMAL": 1.0, "WIDE": 1.8}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_spread_multiplier(self, vol_norm: float, inv_norm: float) -> float:
        """
        Run Mamdani inference and return the defuzzified spread multiplier.

        Parameters
        ----------
        vol_norm : float  in [0, 1]
        inv_norm : float  in [0, 1]

        Returns
        -------
        float : spread multiplier
        """
        vol_norm = float(max(0.0, min(1.0, vol_norm)))
        inv_norm = float(max(0.0, min(1.0, inv_norm)))

        vol_mf = self._volatility_mf(vol_norm)
        inv_mf = self._inventory_mf(inv_norm)

        weighted_sum = 0.0
        weight_total = 0.0

        for (v_label, i_label), out_label in self._RULES.items():
            activation    = min(vol_mf[v_label], inv_mf[i_label])
            weighted_sum += activation * self._CENTROIDS[out_label]
            weight_total += activation

        if weight_total < 1e-9:
            return 1.0

        multiplier = weighted_sum / weight_total
        logger.debug(
            "FuzzySpread | vol=%.3f inv=%.3f => multiplier=%.3f",
            vol_norm, inv_norm, multiplier,
        )
        return multiplier
