"""
Manage alert states based on risk scores.
"""

from alerts.alert_level import AlertLevel
from models.types import RiskScores
from config.settings import(
    WATCH_THRESHOLD,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD
)


class AlertManager:
    """
    Manage alert levels for tracked people.
    """

    def __init__(self) -> None:
        """
        Initialize the alert manager.
        """

        self._alert_levels: dict[int, AlertLevel] = {}


    def get_alert_level(self, risk: float) -> AlertLevel:
        """
        Convert a risk score into
        an alert level.
        """

        if risk >= CRITICAL_THRESHOLD:
            return AlertLevel.CRITICAL

        if risk >= WARNING_THRESHOLD:
            return AlertLevel.WARNING

        if risk >= WATCH_THRESHOLD:
            return AlertLevel.WATCH

        return AlertLevel.NORMAL
    

    def update(self, risk_scores: RiskScores) -> dict[int, AlertLevel]:
        """
        Update alert levels for all tracked people.
        """