"""
Data model representing a detected person.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.types import BoundingBox


@dataclass(slots=True)
class PersonDetection:
    """
    Represents a person detected in a single video frame.
    """

    bbox: BoundingBox
    confidence: float