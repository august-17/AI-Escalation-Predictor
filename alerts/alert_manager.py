"""Stateful alert confirmation with hysteresis and testable timing."""

from __future__ import annotations

import time
from typing import Callable

from alerts.alert_event import AlertEvent
from alerts.alert_level import AlertLevel
from alerts.alert_state import AlertState
from alerts.alert_transition import AlertTransition
from config.settings import (
    ALERT_HYSTERESIS,
    CRITICAL_CONFIRMATION_TIME,
    CRITICAL_THRESHOLD,
    WARNING_CONFIRMATION_TIME,
    WARNING_THRESHOLD,
    WATCH_CONFIRMATION_TIME,
    WATCH_THRESHOLD,
)
from models.risk_breakdown import RiskBreakdown
from models.types import RiskScores

_LEVEL_ORDER = {
    AlertLevel.NORMAL: 0,
    AlertLevel.WATCH: 1,
    AlertLevel.WARNING: 2,
    AlertLevel.CRITICAL: 3,
}


class AlertManager:
    """Convert temporal cue scores into confirmed, non-flapping alert states."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._states: dict[int, AlertState] = {}
        self._events: list[AlertEvent] = []
        self._pending_events: list[AlertEvent] = []

    @staticmethod
    def _base_level(risk: float) -> AlertLevel:
        if risk >= CRITICAL_THRESHOLD:
            return AlertLevel.CRITICAL
        if risk >= WARNING_THRESHOLD:
            return AlertLevel.WARNING
        if risk >= WATCH_THRESHOLD:
            return AlertLevel.WATCH
        return AlertLevel.NORMAL

    def _level_with_hysteresis(self, risk: float, previous: AlertLevel) -> AlertLevel:
        candidate = self._base_level(risk)
        if _LEVEL_ORDER[candidate] >= _LEVEL_ORDER[previous]:
            return candidate
        if previous is AlertLevel.CRITICAL and risk >= CRITICAL_THRESHOLD - ALERT_HYSTERESIS:
            return AlertLevel.CRITICAL
        if previous is AlertLevel.WARNING and risk >= WARNING_THRESHOLD - ALERT_HYSTERESIS:
            return AlertLevel.WARNING
        if previous is AlertLevel.WATCH and risk >= WATCH_THRESHOLD - ALERT_HYSTERESIS:
            return AlertLevel.WATCH
        return candidate

    @staticmethod
    def _transition(previous: AlertLevel, current: AlertLevel) -> AlertTransition:
        if previous is current:
            return AlertTransition.NONE
        if _LEVEL_ORDER[current] > _LEVEL_ORDER[previous]:
            return {
                AlertLevel.WATCH: AlertTransition.ENTER_WATCH,
                AlertLevel.WARNING: AlertTransition.ENTER_WARNING,
                AlertLevel.CRITICAL: AlertTransition.ENTER_CRITICAL,
            }[current]
        return {
            AlertLevel.WARNING: AlertTransition.DEESCALATE_TO_WARNING,
            AlertLevel.WATCH: AlertTransition.DEESCALATE_TO_WATCH,
            AlertLevel.NORMAL: AlertTransition.RETURN_TO_NORMAL,
        }[current]

    @staticmethod
    def _confirmation_time(level: AlertLevel) -> float:
        return {
            AlertLevel.NORMAL: 0.0,
            AlertLevel.WATCH: WATCH_CONFIRMATION_TIME,
            AlertLevel.WARNING: WARNING_CONFIRMATION_TIME,
            AlertLevel.CRITICAL: CRITICAL_CONFIRMATION_TIME,
        }[level]

    def update(
        self,
        risk_scores: RiskScores,
        breakdowns: dict[int, RiskBreakdown] | None = None,
        timestamp: float | None = None,
    ) -> dict[int, AlertState]:
        now = self._clock() if timestamp is None else timestamp
        active_ids = set(risk_scores)
        for track_id, unclamped_risk in risk_scores.items():
            risk = max(0.0, min(float(unclamped_risk), 1.0))
            state = self._states.get(track_id)
            if state is None:
                state = AlertState(track_id=track_id, entered_at=now)
                self._states[track_id] = state
            level = self._level_with_hysteresis(risk, state.level)
            transition = self._transition(state.level, level)
            state.transition = transition
            state.current_risk = risk
            if level is not state.level:
                state.level = level
                state.entered_at = now
                state.confirmed = level is AlertLevel.NORMAL
                state.event_created = level is AlertLevel.NORMAL
            elif level is AlertLevel.NORMAL:
                state.confirmed = True
                state.event_created = True
            else:
                state.confirmed = now - state.entered_at >= self._confirmation_time(level)

            if state.confirmed and not state.event_created and level is not AlertLevel.NORMAL:
                dominant = "unknown"
                if breakdowns is not None and track_id in breakdowns:
                    dominant = breakdowns[track_id].dominant_factor
                event = AlertEvent(
                    track_id=track_id,
                    level=level,
                    risk=risk,
                    dominant_factor=dominant,
                    timestamp=time.time(),
                )
                self._events.append(event)
                self._pending_events.append(event)
                state.event_created = True

        self._states = {
            track_id: state for track_id, state in self._states.items() if track_id in active_ids
        }
        return self._states.copy()

    def get_new_events(self) -> list[AlertEvent]:
        events, self._pending_events = self._pending_events, []
        return events

    @property
    def events(self) -> tuple[AlertEvent, ...]:
        return tuple(self._events)
