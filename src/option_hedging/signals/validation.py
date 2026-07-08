"""Diagnostics for constructed signal frames."""

from __future__ import annotations

from typing import Any

from finance_project.signals.transforms import _require_numpy, _require_pandas, ensure_panel


def summarize_signal_frame(
    features: Any,
    *,
    asset_col: str = "asset",
    date_col: str = "date",
) -> dict[str, Any]:
    """Return lightweight coverage diagnostics for a feature panel."""

    np = _require_numpy()
    pd = _require_pandas()
    frame = ensure_panel(features, set(), asset_col=asset_col, date_col=date_col)
    signal_cols = [column for column in frame.columns if column not in {asset_col, date_col}]
    summary: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_assets": int(frame[asset_col].nunique()),
        "date_min": frame[date_col].min(),
        "date_max": frame[date_col].max(),
        "n_signals": len(signal_cols),
        "missing_fraction": {},
        "finite_fraction": {},
    }

    for column in signal_cols:
        values = frame[column]
        summary["missing_fraction"][column] = float(values.isna().mean())
        numeric = pd.to_numeric(values.dropna(), errors="coerce").dropna()
        if len(numeric) == 0:
            summary["finite_fraction"][column] = 0.0
        else:
            summary["finite_fraction"][column] = float(np.isfinite(numeric).mean())
    return summary


def audit_signal_frame(
    features: Any,
    *,
    asset_col: str = "asset",
    date_col: str = "date",
    max_missing_fraction: float = 0.95,
) -> list[str]:
    """Return human-readable issues that should be reviewed before modeling."""

    issues: list[str] = []
    summary = summarize_signal_frame(features, asset_col=asset_col, date_col=date_col)

    if summary["n_rows"] == 0:
        issues.append("Feature frame is empty.")
    if summary["n_assets"] == 0:
        issues.append("Feature frame has no assets.")
    if summary["n_signals"] == 0:
        issues.append("Feature frame has no signal columns.")

    for column, missing_fraction in summary["missing_fraction"].items():
        if missing_fraction > max_missing_fraction:
            issues.append(
                f"Signal {column!r} is {missing_fraction:.1%} missing, "
                f"above threshold {max_missing_fraction:.1%}."
            )

    for column, finite_fraction in summary["finite_fraction"].items():
        if finite_fraction < 1.0:
            issues.append(f"Signal {column!r} contains non-finite numeric values.")

    return issues
