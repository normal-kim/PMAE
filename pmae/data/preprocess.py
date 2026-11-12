"""Step 4: encoders fit on the observed entries of the train split only.

Categorical columns: categories that are rare among the observed train entries are merged into one
category (fewer than round(n/100) entries if n/100 > 30, else fewer than 30, with n the number of
observed train entries of the column), then integer encoded. Numerical columns: quantile transform
to a standard normal.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .schema import Schema

OTHER = "<other>"  # the merged category


def merge_threshold(n_observed: int) -> int:
    return round(n_observed / 100) if n_observed / 100 > 30 else 30


def _sorted_labels(labels) -> list[str]:
    """Numeric order when every label is a number, else lexicographic."""
    try:
        return sorted(labels, key=float)
    except ValueError:
        return sorted(labels)


def n_quantiles_for(n_rows: int) -> int:
    return max(min(n_rows // 30, 1000), 10)


class CatVocab:
    """Per column: the frequent observed train categories (sorted), then the merged category if any were rare.

    A category that is rare or unseen in the observed train entries maps to the merged category,
    or to -1 when the column has none.
    """

    def fit(self, X: np.ndarray, M: np.ndarray) -> "CatVocab":
        self.vocab_ = []
        for j in range(X.shape[1]):
            counts = Counter(X[M[:, j], j].tolist())
            t = merge_threshold(int(M[:, j].sum()))
            frequent = [c for c, n in counts.items() if n >= t]
            self.vocab_.append(_sorted_labels(frequent) + ([OTHER] if len(frequent) < len(counts) else []))
        return self

    @property
    def n_categories(self) -> list[int]:
        return [len(v) for v in self.vocab_]

    def codes(self, X: np.ndarray) -> np.ndarray:
        """Integer codes of every entry (-1 where the category cannot be represented)."""
        out = np.full(X.shape, -1, dtype=np.int64)
        for j, vocab in enumerate(self.vocab_):
            lookup = {v: i for i, v in enumerate(vocab)}
            fallback = lookup.get(OTHER, -1)
            out[:, j] = [lookup.get(v, fallback) for v in X[:, j]]
        return out

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        out = np.empty(codes.shape, dtype=object)
        for j, vocab in enumerate(self.vocab_):
            out[:, j] = np.asarray(vocab, dtype=object)[codes[:, j]]
        return out


class NumQuantile:
    """Quantile transform to a standard normal; NaN entries are ignored when fitting and kept as NaN."""

    def fit(self, X: np.ndarray) -> "NumQuantile":
        self.qt_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles_for(X.shape[0]),
            subsample=int(1e9),
            random_state=42,
        ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.qt_.inverse_transform(Z)


class Encoder:
    """Maps the features of a data-space table to the model space every imputer receives:
    quantile-normal numerical columns, then integer codes of the categorical columns (as floats),
    with NaN where an entry is missing."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.d_num = len(schema.num_cols)

    def fit(self, df: pd.DataFrame, M: np.ndarray) -> "Encoder":
        """``M`` (n x d, True = observed) is over ``schema.features``."""
        s, dn = self.schema, self.d_num
        self.num_ = NumQuantile().fit(self._num(df, M)) if dn else None
        self.cat_ = CatVocab().fit(df[list(s.cat_cols)].to_numpy(object), M[:, dn:]) if s.cat_cols else None
        return self

    @property
    def categorical(self) -> list[int]:
        return list(range(self.d_num, len(self.schema.features)))

    @property
    def n_categories(self) -> list[int]:
        return self.cat_.n_categories if self.cat_ else []

    def _num(self, df, M):
        return np.where(M[:, :self.d_num], df[list(self.schema.num_cols)].to_numpy(np.float64), np.nan)

    def transform(self, df: pd.DataFrame, M: np.ndarray) -> np.ndarray:
        """Model-space table with NaN at missing entries (and at categories that cannot be represented)."""
        parts = []
        if self.d_num:
            parts.append(self.num_.transform(self._num(df, M)))
        if self.cat_:
            codes = self.cat_.codes(df[list(self.schema.cat_cols)].to_numpy(object)).astype(np.float64)
            parts.append(np.where(M[:, self.d_num:] & (codes >= 0), codes, np.nan))
        return np.hstack(parts)

    def cat_codes(self, df: pd.DataFrame) -> np.ndarray:
        """Codes of the complete categorical values (the ground truth for accuracy)."""
        return self.cat_.codes(df[list(self.schema.cat_cols)].to_numpy(object))

    def inverse_transform(self, Z: np.ndarray) -> pd.DataFrame:
        """Model-space table (categorical codes already rounded to valid integers) back to data space."""
        s, dn = self.schema, self.d_num
        cols = {}
        if dn:
            cols |= dict(zip(s.num_cols, self.num_.inverse_transform(Z[:, :dn]).T))
        if self.cat_:
            cols |= dict(zip(s.cat_cols, self.cat_.inverse_transform(Z[:, dn:].astype(np.int64)).T))
        return pd.DataFrame(cols)[list(s.features)]
