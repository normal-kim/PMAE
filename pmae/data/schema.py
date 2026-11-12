"""Column roles of a cleaned dataset."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

TASKS = ("binclass", "multiclass", "regression")


@dataclass(frozen=True)
class Schema:
    """Roles of the columns of a complete table.

    The features (numerical first, then categorical) are masked and imputed; the
    target is never masked and is used only for the downstream Utility check.
    Numerical columns are float, categorical columns and a classification target are strings.
    """

    name: str
    task: str
    num_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]
    target: str

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        roles = list(self.num_cols) + list(self.cat_cols) + [self.target]
        if len(set(roles)) != len(roles):
            raise ValueError("num_cols, cat_cols and target must not overlap")

    @property
    def features(self) -> tuple[str, ...]:
        return self.num_cols + self.cat_cols

    @property
    def columns(self) -> tuple[str, ...]:
        return self.features + (self.target,)

    @property
    def is_regression(self) -> bool:
        return self.task == "regression"

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path) -> "Schema":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
