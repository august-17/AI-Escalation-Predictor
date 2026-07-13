"""
Common type aliases used throughout the project.
"""

from typing import TypeAlias

from models.risk_breakdown import RiskBreakdown

BoundingBox: TypeAlias = tuple[int, int, int, int]

TrackPair: TypeAlias = tuple[int, int]

RiskScores: TypeAlias = dict[int, float]

RiskComputationResult: TypeAlias = tuple[RiskScores, dict[int, RiskBreakdown]]