"""Imputers from other packages: EM, GAIN and HyperImpute (hyperimpute), MiceForest (miceforest).

Settings follow Appendix A of the paper. Every imputer is fit on the train split and applied to new rows; the
hyperimpute plugins refit on every table they impute, so EM, GAIN and HyperImpute are wrapped here to do the same.
EM and HyperImpute receive min-max scaled columns (GAIN scales them itself).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simple import MinMax


class EM:
    """EM (Dempster et al., 1977) with hyperimpute's EM steps: a Gaussian mean and covariance are estimated on the
    fit data, and any rows are imputed by their conditional means given their observed entries."""

    def __init__(self, maxit: int = 25, convergence_threshold: float = 1e-6):
        self.maxit, self.convergence_threshold = maxit, convergence_threshold

    def fit(self, X: np.ndarray, categorical=()) -> "EM":
        from hyperimpute.plugins.imputers.plugin_EM import EM as HyperImputeEM

        em = HyperImputeEM(self.maxit, self.convergence_threshold)
        self.scaler_ = MinMax().fit(X)
        Xr = self.scaler_.transform(X)
        observed = ~np.isnan(Xr)
        cols = np.arange(1, Xr.shape[1] + 1)
        mu, sigma = np.nanmean(Xr, 0), np.diag(np.nanvar(Xr, 0))  # hyperimpute's starting point
        self.mu_, self.sigma_ = mu, sigma
        for _ in range(self.maxit):
            try:
                mu_new, sigma_new, Xr = em._em(Xr, mu, sigma, cols * observed - 1, cols * ~observed - 1)
            except np.linalg.LinAlgError:  # singular covariance: keep the last estimates, as hyperimpute does
                break
            self.mu_, self.sigma_ = mu, sigma  # the estimates the last imputation used
            if em._converged(mu, sigma, mu_new, sigma_new):
                break
            mu, sigma = mu_new, sigma_new
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        out, missing = Xs.copy(), np.isnan(Xs)
        for i in np.flatnonzero(missing.any(1)):
            m = missing[i]
            o = ~m
            try:
                out[i, m] = self.mu_[m] + self.sigma_[np.ix_(m, o)] @ np.linalg.inv(self.sigma_[np.ix_(o, o)]) @ (
                    Xs[i, o] - self.mu_[o])
            except np.linalg.LinAlgError:
                out[i, m] = self.mu_[m]
        return self.scaler_.inverse(out)


class HyperImpute:
    """HyperImpute (Jarrett et al., ICML 2022) fit on the fit data. New rows start from the fitted column means and
    are imputed by applying the fitted column models in rounds until they settle, as chained equations do."""

    SETTINGS = dict(optimizer="hyperband",
                    classifier_seed=["logistic_regression", "catboost", "xgboost", "random_forest"],
                    regression_seed=["linear_regression", "catboost_regressor", "xgboost_regressor",
                                     "random_forest_regressor"])

    def __init__(self, seed: int = 0, max_rounds: int = 40):
        self.seed, self.max_rounds = seed, max_rounds

    def _encode(self, X: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(self.scaler_.transform(X))
        df.columns = df.columns.map(str)
        for col, encoder in self.model_.encoders.items():
            known = df[col].notna()
            codes = np.searchsorted(encoder.classes_, df.loc[known, col].to_numpy())
            df.loc[known, col] = np.clip(codes, 0, len(encoder.classes_) - 1)
        return df

    def fit(self, X: np.ndarray, categorical=()) -> "HyperImpute":
        from hyperimpute.plugins.imputers import Imputers

        self.scaler_ = MinMax().fit(X)
        plugin = Imputers().get("hyperimpute", random_state=self.seed, **self.SETTINGS)
        self.fit_data_ = X
        self.fit_out_ = self.scaler_.inverse(np.asarray(plugin.fit_transform(pd.DataFrame(self.scaler_.transform(X))),
                                                        dtype=np.float64))
        self.model_ = plugin.model
        self.start_ = self._encode(X).mean()  # hyperimpute's starting fill: the column means
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X is self.fit_data_:
            return self.fit_out_
        m = self.model_
        df = self._encode(X)
        missing = df.isna()
        df = df.fillna(self.start_)
        for _ in range(self.max_rounds):
            before = df.to_numpy(copy=True)
            for col in m.imputation_order:
                rows = missing[col].to_numpy()
                if not rows.any():
                    continue
                low, high = m.limits[col]
                if col not in m.column_to_model or low == high:
                    df.loc[rows, col] = low if low == high else self.start_[col]
                    continue
                pred = m.column_to_model[col].predict(df.loc[rows, m._get_neighbors_for_col(col)])
                df.loc[rows, col] = np.clip(np.asarray(pred, dtype=np.float64).reshape(-1), low, high)
            if np.abs(df.to_numpy() - before).max() < 1e-8:
                break
        for col, encoder in m.encoders.items():
            df[col] = encoder.inverse_transform(df[col].round().astype(int))
        return self.scaler_.inverse(df.to_numpy(np.float64))


class GAIN:
    """GAIN (Yoon et al., ICML 2018) with hyperimpute's implementation, trained once on the fit data and applied to
    new rows, with the training length, batch size and hint rate of the GAIN paper (hyperimpute fixes the loss
    weight at 10)."""

    def __init__(self, seed: int = 0, batch_size: int = 128, iterations: int = 10000, hint_rate: float = 0.9):
        self.seed = seed
        self.settings = dict(batch_size=batch_size, n_epochs=iterations, hint_rate=hint_rate)

    def fit(self, X: np.ndarray, categorical=()) -> "GAIN":
        import torch
        from hyperimpute.plugins.imputers.plugin_gain import GainImputation
        from hyperimpute.utils.distributions import enable_reproducible_results

        enable_reproducible_results(self.seed)
        self.model_ = GainImputation(**self.settings).fit(torch.tensor(X, dtype=torch.float32))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch
        from hyperimpute.utils.distributions import enable_reproducible_results

        enable_reproducible_results(self.seed)
        out = self.model_.transform(torch.tensor(X, dtype=torch.float32))
        out = np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=np.float64)
        return np.where(np.isnan(X), out, X)


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
