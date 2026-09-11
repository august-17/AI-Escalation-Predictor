"""Explainable feature contributions for one tracked person."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RiskBreakdown:
    """Weighted feature contributions; all values are in the 0–1 score space."""

    proximity: float = 0.0
    approach_speed: float = 0.0
    movement: float = 0.0
    hand_speed: float = 0.0
    arm_extension: float = 0.0
    interaction_duration: float = 0.0
    multi_cue_bonus: float = 0.0
    pose_confidence: float = 0.0
    raw_total: float = 0.0
    smoothed_total: float = 0.0

    def contributions(self) -> dict[str, float]:
        values = asdict(self)
        return {
            key: value
            for key, value in values.items()
            if key not in {"pose_confidence", "raw_total", "smoothed_total"}
        }

    @property
    def dominant_factor(self) -> str:
        contributions = self.contributions()
        return max(contributions, key=contributions.get) if any(contributions.values()) else "none"
