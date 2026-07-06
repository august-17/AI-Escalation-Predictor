"""
Data model representing a tracked person.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.pose_result import PoseResult
from models.types import BoundingBox


@dataclass(slots=True)
class TrackedPerson:
    """
    Represents a person currently being tracked across video frames.
    """

    track_id: int
    bbox: BoundingBox
    confidence: float
    pose: PoseResult | None = None