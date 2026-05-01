"""Leak-safe feature scaling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class FeatureScaler:
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str]) -> "FeatureScaler":
        means = {column: float(frame[column].mean()) for column in columns}
        stds = {column: max(float(frame[column].std()), 1e-6) for column in columns}
        return cls(means=means, stds=stds)

    def transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        transformed = frame.copy()
        for column in columns:
            transformed[column] = (transformed[column] - self.means[column]) / self.stds[column]
        return transformed

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, float]]) -> "FeatureScaler":
        return cls(means=payload["means"], stds=payload["stds"])
