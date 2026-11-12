"""Naive (mean / mode) and KNN imputation."""
from __future__ import annotations

import numpy as np


class Naive:
    """Train mean for numerical columns, train mode for categorical columns."""

    def fit(self, X: np.ndarray, categorical=()) -> "Naive":
        self.fill_ = np.nanmean(X, 0)
        for j in categorical:
            values, counts = np.unique(X[~np.isnan(X[:, j]), j], return_counts=True)
            self.fill_[j] = values[counts.argmax()]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), self.fill_, X)


class MinMax:
    """[0, 1] scaling with the observed train range, for the methods that are not scale-free."""

    def fit(self, X: np.ndarray) -> "MinMax":
        self.lo_ = np.nanmin(X, 0)
        self.range_ = np.nanmax(X, 0) - self.lo_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.lo_) / self.range_

    def inverse(self, Z):
        return Z * self.range_ + self.lo_


class KNN:
    """scikit-learn's KNNImputer (5 neighbors) on min-max scaled columns."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, categorical=()) -> "KNN":
        from sklearn.impute import KNNImputer

        self.scaler_ = MinMax().fit(X)
        self.knn_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.scaler_.transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.inverse(self.knn_.transform(self.scaler_.transform(X)))
