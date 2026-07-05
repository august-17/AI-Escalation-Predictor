"""
Data model representing a single pose landmark.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PoseLandmark:
    """
    Represents one pose landmark in image coordinates.
    """

    x: int
    y: int
    visibility: float