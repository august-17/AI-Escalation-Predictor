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

        self._pending_events: list[AlertEvent] = []


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

            state = self._get_or_create_state(track_id, level)

            self._update_state(state, level)

            self._update_confirmation(state)

            if (
                state.confirmed
                and not state.event_created
            ):
                self._create_event(state)
                state.event_created = True

        self._remove_inactive_states(active_ids)

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

        self._pending_events.append(event)


    def get_new_events(self) -> list[AlertEvent]:
        """
        Return newly created alert events and clear the pending queue.
        """

        events = self._pending_events.copy()
        self._pending_events.clear()

        return events
    

    def _get_or_create_state(self, track_id: int, level: AlertLevel) -> AlertState:
        """
        Retrieve an existing alert state or create a new one.
        """

        state = self._alert_states.get(track_id)

        if state is None:
            state = AlertState(
                track_id=track_id,
                level=level
            )
            self._alert_states[track_id] = state

        return state
    

    def _update_state(self, state: AlertState, level: AlertLevel) -> None:
        """
        Update the alert state for a person.
        """
        previous_level = state.level
        transition = self._get_transition(previous_level, level)
        state.transition = transition

        if transition != AlertTransition.NONE:
            state.entered_at = time.time()
            state.event_created = False

        state.level = level


    def _remove_inactive_states(self, active_ids: set[int]) -> None:
        """
        Remove states for people that are no longer tracked.
        """
        self._alert_states = {
            track_id: state
            for track_id, state
            in self._alert_states.items()
            if track_id in active_ids
        }

