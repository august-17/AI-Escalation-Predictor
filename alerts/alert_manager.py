"""
Manage alert states based on risk scores.
"""
import time

from alerts.alert_level import AlertLevel
from alerts.alert_state import AlertState
from alerts.alert_transition import AlertTransition
from alerts.alert_event import AlertEvent
from models.types import RiskScores
from config.settings import(
    WATCH_THRESHOLD,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD,
    WATCH_CONFIRMATION_TIME,
    WARNING_CONFIRMATION_TIME,
    CRITICAL_CONFIRMATION_TIME
)


class AlertManager:
    """
    Manage alert levels for tracked people.
    """

    def __init__(self) -> None:
        """
        Initialize the alert manager.
        """

        self._alert_states: dict[int, AlertState] = {}

        self._events: list[AlertEvent] = []


    def _compute_alert_level(self, risk: float) -> AlertLevel:
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
    

    def update(self, risk_scores: RiskScores) -> dict[int, AlertState]:
        """
        Update alert levels for all tracked people.
        """

        active_ids: set[int] = set()

        for track_id, risk in risk_scores.items():

            active_ids.add(track_id)

            level = self._compute_alert_level(risk)

            state = self._alert_states.get(track_id)

            if state is None:
                state = AlertState(
                    track_id=track_id,
                    level=level
                )
                self._alert_states[track_id] = state
            else:
                previous_level = state.level
                transition = self._get_transition(previous_level, level)
                state.transition = transition

                if previous_level != level:
                    state.entered_at = time.time()

                state.level = level

            self._update_confirmation(state)

        self._alert_states = {
            track_id: state
            for track_id, state
            in self._alert_states.items()
            if track_id in active_ids
        }

        return self._alert_states.copy()
    
    
    def _get_transition(
        self,
        previous: AlertLevel,
        current: AlertLevel
    ) -> AlertTransition:
        """
        Determine the transition between two alert levels.
        """

        if previous == current:
            return AlertTransition.NONE

        transitions = {
            (AlertLevel.NORMAL, AlertLevel.WATCH):
                AlertTransition.ENTER_WATCH,

            (AlertLevel.WATCH, AlertLevel.WARNING):
                AlertTransition.ENTER_WARNING,

            (AlertLevel.WARNING, AlertLevel.CRITICAL):
                AlertTransition.ENTER_CRITICAL,

            (AlertLevel.CRITICAL, AlertLevel.WARNING):
                AlertTransition.DEESCALATE_TO_WARNING,

            (AlertLevel.WARNING, AlertLevel.WATCH):
                AlertTransition.DEESCALATE_TO_WATCH,

            (AlertLevel.WATCH, AlertLevel.NORMAL):
                AlertTransition.RETURN_TO_NORMAL
        }

        return transitions.get(
            (previous, current),
            AlertTransition.NONE
        )
    

    def _confirmation_time(self, level: AlertLevel) -> float:
        """
        Return the confirmation duration
        for an alert level.
        """

        if level == AlertLevel.WATCH:
            return WATCH_CONFIRMATION_TIME

        if level == AlertLevel.WARNING:
            return WARNING_CONFIRMATION_TIME

        if level == AlertLevel.CRITICAL:
            return CRITICAL_CONFIRMATION_TIME

        return 0.0
    

    def _update_confirmation(self, state: AlertState) -> None:

        """
        Update the confirmation status of an alert state.
        """

        if state.level == AlertLevel.NORMAL:
            state.confirmed = False
            return

        duration = time.time() - state.entered_at

        state.confirmed = duration >= self._confirmation_time(state.level)


    def _create_event(self, state: AlertState) -> None:
        """
        Create and store a new alert event.
        """

        event = AlertEvent(
            track_id=state.track_id,
            level=state.level,
        )

        self._events.append(event)

