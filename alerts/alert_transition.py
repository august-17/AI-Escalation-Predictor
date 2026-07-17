from enum import Enum


class AlertTransition(Enum):
    NONE = "None"

    ENTER_WATCH = "Enter Watch"
    ENTER_WARNING = "Enter Warning"
    ENTER_CRITICAL = "Enter Critical"

    DEESCALATE_TO_WARNING = "De-escalate to Warning"
    DEESCALATE_TO_WATCH = "De-escalate to Watch"
    RETURN_TO_NORMAL = "Return to Normal"