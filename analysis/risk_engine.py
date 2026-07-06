"""
Computes an escalation risk score for tracked people.
"""

from __future__ import annotations

from models.tracked_person import TrackedPerson


class RiskEngine:
    """
    Computes an escalation score.

    Current implementation is a placeholder that always
    returns zero until behavioral features are added.
    """

    @staticmethod
    def compute(people: list[TrackedPerson]) -> dict[int, float]:
        """
        Compute a risk score for each tracked person.

        Args:
            people: List of tracked people.

        Returns:
            Dictionary mapping track IDs to risk scores.
        """

        return {
            person.track_id: 0.0
            for person in people
        }