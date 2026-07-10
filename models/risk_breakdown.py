"""
Stores the contribution of each behavioral feature
to a person's overall risk score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskBreakdown:
    """
    Detailed breakdown of a person's risk score.
    """

    proximity: float = 0.0
    movement: float = 0.0
    hand_speed: float = 0.0
    raw_total: float = 0.0
    smoothed_total: float = 0.0