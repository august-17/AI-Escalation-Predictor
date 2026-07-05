"""
Data model representing a tracked person.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackedPerson:
    """
    Represents a tracked person.
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float