"""High-level signal-library API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from finance_project.signals.base import SignalRegistry, SignalSpec
from finance_project.signals.definitions import make_default_signal_specs
from finance_project.signals.transforms import ensure_panel

MissingPolicy = Literal["raise", "skip"]


class SignalLibrary:
    """Compute a registered collection of financial signals on a panel."""

    def __init__(self, specs: Iterable[SignalSpec]) -> None:
        self.registry = SignalRegistry(specs)

    @classmethod
    def default(cls) -> "SignalLibrary":
        return cls(make_default_signal_specs())

    def catalog(self) -> list[dict[str, Any]]:
        return self.registry.catalog()

    def compute(
        self,
        data: Any,
        *,
        names: Iterable[str] | None = None,
        families: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        asset_col: str = "asset",
        date_col: str = "date",
        on_missing: MissingPolicy = "raise",
    ) -> Any:
        """Compute selected signals and return one feature frame.

        Parameters
        ----------
        data:
            Asset-date panel.
        names, families, tags:
            Optional selectors. When none are provided, all registered signals
            are attempted.
        on_missing:
            ``"raise"`` fails on unavailable raw columns. ``"skip"`` computes
            only signals whose required columns are present and stores skipped
            signal names in ``features.attrs["skipped_signals"]``.
        """

        if on_missing not in {"raise", "skip"}:
            raise ValueError("on_missing must be either 'raise' or 'skip'.")

        selected = self.registry.select(names=names, families=families, tags=tags)
        if not selected:
            raise ValueError("No signals selected.")

        base = ensure_panel(data, set(), asset_col=asset_col, date_col=date_col)
        features = base[[date_col, asset_col]].drop_duplicates().copy()
        skipped: list[dict[str, str]] = []

        for spec in selected:
            missing_required = sorted(set(spec.required_columns).difference(data.columns))
            if missing_required:
                if on_missing == "skip":
                    skipped.append(
                        {
                            "name": spec.name,
                            "reason": f"Missing required column(s): {', '.join(missing_required)}",
                        }
                    )
                    continue
                raise KeyError(f"Missing required column(s): {', '.join(missing_required)}")

            result = spec(data, asset_col=asset_col, date_col=date_col)

            missing_outputs = set(spec.output_columns).difference(result.columns)
            if missing_outputs:
                missing = ", ".join(sorted(missing_outputs))
                raise ValueError(f"Signal {spec.name!r} did not return output column(s): {missing}")

            result = result[[date_col, asset_col, *spec.output_columns]]
            duplicate_mask = result.duplicated([asset_col, date_col])
            if bool(duplicate_mask.any()):
                raise ValueError(f"Signal {spec.name!r} returned duplicate asset-date rows.")

            features = features.merge(result, on=[date_col, asset_col], how="left", validate="one_to_one")

        if len(features.columns) == 2:
            skipped_names = ", ".join(item["name"] for item in skipped)
            raise ValueError(f"No signals were computed. Skipped: {skipped_names}")

        features.attrs["skipped_signals"] = skipped
        skipped_name_set = {item["name"] for item in skipped}
        features.attrs["signal_names"] = [
            spec.name for spec in selected if spec.name not in skipped_name_set
        ]
        return features

    @property
    def names(self) -> tuple[str, ...]:
        return self.registry.names

    @property
    def families(self) -> tuple[str, ...]:
        return self.registry.families

    def __len__(self) -> int:
        return len(self.registry)
