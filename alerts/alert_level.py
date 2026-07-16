"""
Alert level definitions.
"""

from enum import Enum


class AlertLevel(Enum):
    """
    Alert severity levels.
    """

    NORMAL = "Normal"

    WATCH = "Watch"

    WARNING = "Warning"

    CRITICAL = "Critical"