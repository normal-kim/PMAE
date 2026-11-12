"""Imputers from other packages: EM, GAIN and HyperImpute (hyperimpute), MiceForest (miceforest).

Settings follow Appendix A of the paper. EM, GAIN and HyperImpute receive min-max scaled columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simple import MinMax

HYPERIMPUTE = {
    "em": ("EM", dict(maxit=25, convergence_threshold=1e-6)),
    "gain": ("gain", {}),
    "hyperimpute": ("hyperimpute", dict(
        optimizer="hyperband",
        classifier_seed=["logistic_regression", "catboost", "xgboost", "random_forest"],
        regression_seed=["linear_regression", "catboost_regressor", "xgboost_regressor", "random_forest_regressor"],
    )),
}


class HyperImputePlugin:
    def __init__(self, name: str, seed: int = 0):
        self.name, self.seed = name, seed

    def fit(self, X: np.ndarray, categorical=()) -> "HyperImputePlugin":
        from hyperimpute.plugins.imputers import Imputers

        plugin, kwargs = HYPERIMPUTE[self.name]
        self.scaler_ = MinMax().fit(X)
        self.plugin_ = Imputers().get(plugin, random_state=self.seed, **kwargs)
        self.plugin_.fit(pd.DataFrame(self.scaler_.transform(X)))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = self.plugin_.transform(pd.DataFrame(self.scaler_.transform(X)))
        return self.scaler_.inverse(np.asarray(out, dtype=np.float64))


class MiceForest:
    """miceforest (LightGBM) with 5 imputed datasets: the mean for numerical columns, the mode for categorical.

    Categorical columns use miceforest's "fast" mean matching (a class drawn from the predicted probabilities):
    the default matches on log-odds, which are infinite when LightGBM predicts a probability of 0 or 1.
    """

    def __init__(self, datasets: int = 5, iterations: int = 5, seed: int = 0):
        self.datasets, self.iterations, self.seed = datasets, iterations, seed

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=[f"c{j}" for j in range(X.shape[1])])
        for j in self.categorical_:
            df[f"c{j}"] = pd.Categorical(df[f"c{j}"], categories=self.levels_[j])
        return df

    def _aggregate(self, imputed) -> np.ndarray:
        # explicit iteration: impute_new_data reports 0 iterations, so the default returns the initial draws
        draws = np.stack([imputed.complete_data(k, iteration=self.iterations).to_numpy(np.float64)
                          for k in range(self.datasets)])
        out = draws.mean(0)
        for j in self.categorical_:
            col = draws[:, :, j].astype(np.int64)
            counts = np.stack([(col == c).sum(0) for c in range(int(col.max()) + 1)])
            out[:, j] = counts.argmax(0)
        return out

    def fit(self, X: np.ndarray, categorical=()) -> "MiceForest":
        import miceforest as mf

        self.categorical_ = list(categorical)
        self.levels_ = {j: np.unique(X[~np.isnan(X[:, j]), j]) for j in self.categorical_}
        self.kernel_ = mf.ImputationKernel(self._frame(X), num_datasets=self.datasets, random_state=self.seed,
                                           mean_match_strategy={f"c{j}": "fast" for j in self.categorical_})
        self.kernel_.mice(self.iterations)
        self.fit_data_ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X is self.fit_data_:
            return self._aggregate(self.kernel_)
        return self._aggregate(self.kernel_.impute_new_data(self._frame(X), random_state=self.seed))
