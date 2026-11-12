"""Steps 2-4 for one (dataset, pattern, mechanism, seed) scenario, and the map back to data space (step 6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets import load_dataset
from .patterns import MASKED_SPLITS, make_masks, propensity_features, setting_name
from .preprocess import Encoder, NumQuantile
from .schema import Schema
from .split import SPLITS, split_indices


@dataclass
class Scenario:
    schema: Schema
    pattern: str
    mechanism: str
    p_col: float
    seed: int
    tables: dict[str, pd.DataFrame]  # complete data-space splits (features and target)
    masks: dict[str, np.ndarray]  # observed masks over schema.features, True = observed (train, val)
    encoder: Encoder  # fit on the observed train entries
    X: dict[str, np.ndarray]  # model space, NaN where missing (train, val)
    evaluator: NumQuantile | None  # numerical scale for the metrics, fit on the complete train split

    @property
    def name(self) -> str:
        return setting_name(self.pattern, self.mechanism, self.p_col)

    @property
    def categorical(self) -> list[int]:
        return self.encoder.categorical

    @property
    def n_categories(self) -> list[int]:
        return self.encoder.n_categories

    def finalize(self, split: str, Z: np.ndarray) -> np.ndarray:
        """An imputer's output in model space: observed entries restored, categorical codes rounded and
        clipped to valid categories."""
        X = self.X[split]
        Z = np.where(np.isnan(X), Z, X)
        for k, j in enumerate(self.categorical):
            Z[:, j] = np.clip(np.rint(Z[:, j]), 0, self.n_categories[k] - 1)
        return Z

    def imputed_table(self, split: str, Z: np.ndarray) -> pd.DataFrame:
        """Data-space table of ``split``: observed entries as given, missing entries from ``Z``
        (the output of ``finalize``), plus the complete target."""
        complete = self.tables[split]
        feats = list(self.schema.features)
        decoded = self.encoder.inverse_transform(Z)
        table = complete[feats].where(self.masks[split], decoded)
        table[self.schema.target] = complete[self.schema.target].to_numpy()
        return table.astype(complete.dtypes.to_dict())


def build_scenario(dataset: str, pattern: str, mechanism: str = "mnar", seed: int = 0, p_col: float = 0.6,
                   root=None) -> Scenario:
    df, schema = load_dataset(dataset, root)
    y = None if schema.is_regression else df[schema.target].to_numpy()
    split = split_indices(len(df), y=y, seed=0)
    tables = {s: df.iloc[split[s]].reset_index(drop=True) for s in SPLITS}

    Z, groups = propensity_features({s: tables[s] for s in MASKED_SPLITS}, schema)
    masks = make_masks(Z, groups, pattern, mechanism, seed, p_col)

    encoder = Encoder(schema).fit(tables["train"], masks["train"])
    X = {s: encoder.transform(tables[s], masks[s]) for s in MASKED_SPLITS}
    evaluator = (NumQuantile().fit(tables["train"][list(schema.num_cols)].to_numpy(np.float64))
                 if schema.num_cols else None)
    return Scenario(schema, pattern, mechanism, p_col, seed, tables, masks, encoder, X, evaluator)
