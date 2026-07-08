"""Signal-library public API."""

from option_hedging.signals.base import SignalRegistry, SignalSpec
from option_hedging.signals.definitions import (
    make_default_signal_specs,
    make_hedging_signal_specs,
    passthrough_signal,
)
from option_hedging.signals.library import SignalLibrary
from option_hedging.signals.targets import (
    create_downside_target,
    create_future_realized_volatility_target,
    create_future_return_target,
)
from option_hedging.signals.validation import (
    audit_signal_frame,
    summarize_signal_frame,
)

__all__ = [
    "SignalLibrary",
    "SignalRegistry",
    "SignalSpec",
    "audit_signal_frame",
    "create_downside_target",
    "create_future_realized_volatility_target",
    "create_future_return_target",
    "make_default_signal_specs",
    "make_hedging_signal_specs",
    "passthrough_signal",
    "summarize_signal_frame",
]
