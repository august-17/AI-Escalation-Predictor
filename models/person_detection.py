"""
Data model representing a detected person.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PersonDetection:
    """
    Represents a single detected person.
    """

    bbox: tuple[int, int, int, int]
    confidence: float