"""Step 6: imputation metrics on the missing entries, and Utility.

Imputation (on the entries the pattern made missing):
- R2 of each numerical column (1 - SSE / SST, clipped at 0; 0 when its true missing values are all equal), on a
  quantile-normal scale fit on the complete train split (used by the evaluator only).
- Accuracy of each categorical column (merged categories as the encoder defines them).
- Imp. Acc: the mean of these per-column scores over the columns with missing entries.
- RMSE over all missing numerical entries, on the same scale min-max scaled to [0, 1].

Utility: XGBoost, a random forest and a linear model are trained on the imputed train split, each selected over a
small grid on the imputed validation split, and scored on the complete test split (AUROC, one-vs-rest macro for
multiclass; RMSE of log(clip(y, 1, 2e4)) for regression). Their mean score is reported relative to the same models
trained and selected on the complete splits. The models run on CPU with fixed threads and seeds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder

from .data.scenario import Scenario
from .data.schema import Schema

DOWNSTREAM = ("xgboost", "random_forest", "linear")
GRIDS = {
    "xgboost": {"n_estimators": [10, 50, 100], "min_child_weight": [1, 10], "max_depth": [5, 10, 20],
                "gamma": [0.0, 1.0]},
    "random_forest": {"n_estimators": [100], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 10]},
    "linear": {"C": [0.1, 1.0, 10.0]},  # logistic regression; for regression, ridge with alpha = 1 / C
}


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    """1 - SSE / SST; 0 when the true values are all equal, where R2 is undefined."""
    if np.ptp(true) == 0:
        return 0.0
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def imputation_metrics(scn: Scenario, split: str, Z: np.ndarray) -> dict:
    """``Z``: the finalized model-space table of ``split`` (see Scenario.finalize)."""
    s, dn = scn.schema, len(scn.schema.num_cols)
    true, miss = scn.tables[split], ~scn.masks[split]
    columns, scores = {}, []
    sq_err, n_num = 0.0, 0
    if dn:
        ev = scn.evaluator
        ev_true = ev.transform(true[list(s.num_cols)].to_numpy(np.float64))
        ev_pred = ev.transform(scn.encoder.num_.inverse_transform(Z[:, :dn]))
        ev_train = ev.transform(scn.tables["train"][list(s.num_cols)].to_numpy(np.float64))
        span = ev_train.max(0) - ev_train.min(0)
        span[span == 0] = 1.0
        for j, c in enumerate(s.num_cols):
            mj = miss[:, j]
            if not mj.any():
                continue
            t, p = ev_true[mj, j], ev_pred[mj, j]
            r2 = r2_score(t, p)
            columns[c] = {"r2": r2, "n_missing": int(mj.sum())}
            scores.append(max(r2, 0.0))
            sq_err += (((p - t) / span[j]) ** 2).sum()
            n_num += int(mj.sum())
    if s.cat_cols:
        true_codes = scn.encoder.cat_codes(true)
        pred_codes = Z[:, dn:].astype(np.int64)
        for k, c in enumerate(s.cat_cols):
            mk = miss[:, dn + k]
            if not mk.any():
                continue
            acc = float((pred_codes[mk, k] == true_codes[mk, k]).mean())
            columns[c] = {"acc": acc, "n_missing": int(mk.sum())}
            scores.append(acc)
    r2 = [max(v["r2"], 0.0) for v in columns.values() if "r2" in v]
    acc = [v["acc"] for v in columns.values() if "acc" in v]
    return {
        "imp_acc": float(np.mean(scores)) if scores else float("nan"),
        "r2": float(np.mean(r2)) if r2 else float("nan"),
        "acc": float(np.mean(acc)) if acc else float("nan"),
        "rmse": float(np.sqrt(sq_err / n_num)) if n_num else float("nan"),
        "missing_rate": float(miss.mean()),
        "columns": columns,
    }


def _features(df: pd.DataFrame, schema: Schema, fitted=None):
    """Numerical columns standardized with the statistics of the training table, categorical columns one-hot."""
    num = df[list(schema.num_cols)].to_numpy(np.float64)
    if fitted is None:
        sd = num.std(0)
        fitted = (num.mean(0), np.where(sd > 0, sd, 1.0),
                  {c: OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[[c]].astype(str))
                   for c in schema.cat_cols})
    mean, sd, encoders = fitted
    parts = [(num - mean) / sd] + [encoders[c].transform(df[[c]].astype(str)) for c in schema.cat_cols]
    return np.hstack(parts).astype(np.float32), fitted


def _auc(y, proba, n_classes):
    present = np.unique(y)
    if len(present) < 2:
        return 0.5
    if n_classes == 2:
        return roc_auc_score(y, proba[:, 1])
    p = proba[:, present]
    return roc_auc_score(y, p / p.sum(1, keepdims=True), multi_class="ovr", labels=present)


def _model(kind: str, params: dict, regression: bool, seed: int, n_jobs: int):
    if kind == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        common = {"n_jobs": n_jobs, "tree_method": "hist", "device": "cpu", "random_state": seed}
        if regression:
            return XGBRegressor(objective="reg:squarederror", **params, **common)
        return XGBClassifier(**params, **common)
    if kind == "random_forest":
        return (RandomForestRegressor if regression else RandomForestClassifier)(**params, n_jobs=n_jobs,
                                                                                 random_state=seed)
    if regression:
        return Ridge(alpha=1.0 / params["C"])
    return LogisticRegression(C=params["C"], max_iter=1000)


def downstream_scores(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, schema: Schema, seed: int = 0,
                      n_jobs: int = 4) -> dict[str, float]:
    """Test-split score of each downstream model, fit on ``train`` and selected on ``val``:
    AUROC (classification) or RMSE on log(clip(y, 1, 2e4)) (regression)."""
    X_fit, fitted = _features(train, schema)
    X_val, X_test = _features(val, schema, fitted)[0], _features(test, schema, fitted)[0]
    if schema.is_regression:
        f = lambda v: np.log(np.clip(np.asarray(v, np.float64), 1, 20000))
        y_fit, y_val, y_test = (f(t[schema.target]) for t in (train, val, test))
        score, pick = (lambda m, A, b: mean_squared_error(b, m.predict(A)) ** 0.5), min
    else:
        labels = np.unique(train[schema.target].astype(str))
        y_fit, y_val, y_test = (np.searchsorted(labels, t[schema.target].astype(str)) for t in (train, val, test))
        score, pick = (lambda m, A, b: _auc(b, m.predict_proba(A), len(labels))), max
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for kind in DOWNSTREAM:
            fits = [_model(kind, p, schema.is_regression, seed, n_jobs).fit(X_fit, y_fit)
                    for p in ParameterGrid(GRIDS[kind])]
            out[kind] = float(score(pick(fits, key=lambda m: score(m, X_val, y_val)), X_test, y_test))
    return out


def utility(scores: dict[str, float], reference: dict[str, float], schema: Schema) -> float:
    """Mean downstream score relative to the mean for the complete train table (higher is better for both tasks)."""
    s, r = np.mean([scores[k] for k in DOWNSTREAM]), np.mean([reference[k] for k in DOWNSTREAM])
    return float(r / s if schema.is_regression else s / r)
