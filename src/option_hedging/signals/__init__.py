"""Signal-library public API."""

from finance_project.signals.base import SignalRegistry, SignalSpec
from finance_project.signals.library import SignalLibrary
from finance_project.signals.targets import create_future_return_target
from finance_project.signals.validation import audit_signal_frame, summarize_signal_frame

__all__ = [
    "SignalLibrary",
    "SignalRegistry",
    "SignalSpec",
    "audit_signal_frame",
    "create_future_return_target",
    "summarize_signal_frame",
]

