"""
Data model representing a person's pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.pose_landmark import PoseLandmark


@dataclass(slots=True)
class PoseResult:
    """
    Represents a person's estimated pose.
    """

    landmarks: list[PoseLandmark] = field(default_factory=list)