"""Signal metadata and registry primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol


class SignalComputer(Protocol):
    """Callable contract for a signal transform."""

    def __call__(self, data: Any, *, asset_col: str, date_col: str) -> Any:
        """Return a dataframe with key columns plus the signal output columns."""


@dataclass(frozen=True)
class SignalSpec:
    """Metadata and compute function for one registered signal."""

    name: str
    family: str
    description: str
    required_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    compute: SignalComputer = field(repr=False, compare=False)
    lookback: int | None = None
    tags: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Signal name must be non-empty.")
        if not self.family:
            raise ValueError(f"Signal {self.name!r} must define a family.")
        if not self.output_columns:
            raise ValueError(f"Signal {self.name!r} must define output columns.")
        if self.lookback is not None and self.lookback < 1:
            raise ValueError(f"Signal {self.name!r} has invalid lookback {self.lookback}.")

        object.__setattr__(self, "required_columns", tuple(self.required_columns))
        object.__setattr__(self, "output_columns", tuple(self.output_columns))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable metadata for experiment manifests."""

        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "required_columns": list(self.required_columns),
            "output_columns": list(self.output_columns),
            "lookback": self.lookback,
            "tags": list(self.tags),
            "params": dict(self.params),
        }

    def __call__(self, data: Any, *, asset_col: str, date_col: str) -> Any:
        return self.compute(data, asset_col=asset_col, date_col=date_col)


class SignalRegistry:
    """Ordered collection of signal specs."""

    def __init__(self, specs: Iterable[SignalSpec] = ()) -> None:
        self._specs: dict[str, SignalSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SignalSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate signal name: {spec.name}")
        overlapping_outputs = set(spec.output_columns).intersection(self.output_columns)
        if overlapping_outputs:
            overlap = ", ".join(sorted(overlapping_outputs))
            raise ValueError(f"Duplicate signal output column(s): {overlap}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> SignalSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown signal {name!r}") from exc

    def select(
        self,
        *,
        names: Iterable[str] | None = None,
        families: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[SignalSpec]:
        """Select specs by name, family, and/or tag while preserving order."""

        if names is not None:
            name_list = list(names)
            missing = [name for name in name_list if name not in self._specs]
            if missing:
                raise KeyError(f"Unknown signal(s): {', '.join(missing)}")
            specs = [self._specs[name] for name in name_list]
        else:
            specs = list(self._specs.values())

        if families is not None:
            family_set = set(families)
            specs = [spec for spec in specs if spec.family in family_set]

        if tags is not None:
            tag_set = set(tags)
            specs = [spec for spec in specs if tag_set.intersection(spec.tags)]

        return specs

    def catalog(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self._specs.values()]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(spec.family for spec in self._specs.values()))

    @property
    def output_columns(self) -> tuple[str, ...]:
        outputs: list[str] = []
        for spec in self._specs.values():
            outputs.extend(spec.output_columns)
        return tuple(outputs)

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

