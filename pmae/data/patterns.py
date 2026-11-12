"""Step 3: semi-synthetic missing patterns. Masks use True = observed.

A pattern sets which feature columns get missing values and the observed probability p_j of each:

- monotone: round(p_col * d) columns (at least one), in a random order, with p_j decreasing evenly from 0.8 to 0.2
            along it (0.5 for a single column); the other columns stay complete. The masks are nested: a row
            observed in a column is also observed in every column before it.
- quasi:    every column; round(0.6 * d) of them with p_j ~ U(0.95, 0.99), the rest with p_j ~ U(0.2, 0.8). The
            masks are nested as in monotone (columns from most to least observed), with exceptions: a row missing
            in a column is observed in the next one with 0.2 times the probability it would have if observed.
- general:  every column, p_j ~ U(0.2, 0.8), drawn independently.
- partial-a: round(p_col * d) columns (at least one) with p_j = 0.5, drawn independently; the other columns stay
             complete (the Monotone pattern of the paper's experiments).
- partial-b: the observed probabilities of quasi, drawn independently (the Quasi-Monotone pattern of the paper's
             experiments).

A mechanism sets how missingness depends on the data. For each column j with missing values,
P(x_ij observed) = sigmoid(b_j + z_i' w_j), where z_i encodes the complete values of row i
(quantile-normal numerical, one-hot categorical; encoders fit on the complete train split).
w_j ~ N(0, I) on the allowed features is rescaled so that z' w_j has unit SD on the train split,
and b_j is solved so that the mean observed probability on the train split equals p_j.
For monotone and quasi, this probability is conditional on being observed in the previous column (times the
return factor when missing there), and b_j is solved for the resulting share of rows observed in column j.
The validation split uses the same (w_j, b_j). Allowed features:

- mnar: the features of the columns with missing values, column j included.
- mar:  the features of the complete columns (monotone and partial-a only).
- mcar: none (w_j = 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from .preprocess import n_quantiles_for
from .schema import Schema

PATTERNS = ("monotone", "quasi", "general", "partial-a", "partial-b")
P_COL = ("monotone", "partial-a")  # the patterns where only a share p_col of the columns gets missing values
MECHANISMS = ("mnar", "mar", "mcar")
MASKED_SPLITS = ("train", "val")  # the test split stays complete
QUASI_NEAR_COMPLETE = 0.6
RETURN = {"monotone": 0.0, "quasi": 0.2}  # nested patterns: P(observed after a missing column) / P(after an observed one)


def setting_name(pattern: str, mechanism: str, p_col: float = 0.6) -> str:
    return f"{pattern}{p_col if pattern in P_COL else ''}-{mechanism}"


def propensity_features(tables: dict[str, pd.DataFrame], schema: Schema) -> tuple[dict[str, np.ndarray], list]:
    """Encoded complete features of each split, and the positions of each column's encoded features."""
    parts, owner = {s: [] for s in tables}, []
    num, cat = list(schema.num_cols), list(schema.cat_cols)
    if num:
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles_for(len(tables["train"])),
                                 subsample=int(1e9), random_state=0).fit(tables["train"][num].to_numpy(np.float64))
        for s, t in tables.items():
            parts[s].append(qt.transform(t[num].to_numpy(np.float64)))
        owner += num
    if cat:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tables["train"][cat].to_numpy(object))
        for s, t in tables.items():
            parts[s].append(ohe.transform(t[cat].to_numpy(object)))
        owner += [c for c, cats in zip(cat, ohe.categories_) for _ in cats]
    owner = np.asarray(owner, dtype=object)
    groups = [np.flatnonzero(owner == c) for c in schema.features]
    return {s: np.concatenate(v, axis=1) for s, v in parts.items()}, groups


def observed_probabilities(d: int, pattern: str, rng: np.random.Generator, p_col: float = 0.6) -> np.ndarray:
    """p_j for every column (1.0 for columns that stay complete)."""
    if pattern in P_COL:
        p = np.ones(d)
        cols = rng.choice(d, max(1, round(p_col * d)), replace=False)
        p[cols] = np.linspace(0.8, 0.2, len(cols)) if pattern == "monotone" and len(cols) > 1 else 0.5
    elif pattern in ("quasi", "partial-b"):
        p = rng.uniform(0.2, 0.8, d)
        near = rng.choice(d, round(QUASI_NEAR_COMPLETE * d), replace=False)
        p[near] = rng.uniform(0.95, 0.99, len(near))
    elif pattern == "general":
        p = rng.uniform(0.2, 0.8, d)
    else:
        raise ValueError(f"pattern must be one of {PATTERNS}, got {pattern!r}")
    return p


def make_masks(
    Z: dict[str, np.ndarray],
    groups: list[np.ndarray],
    pattern: str,
    mechanism: str,
    seed: int,
    p_col: float = 0.6,
) -> dict[str, np.ndarray]:
    """Observed-masks (n x d) for the train and validation splits."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}")
    plan_ss, w_ss, *split_ss = np.random.SeedSequence(seed).spawn(2 + len(MASKED_SPLITS))
    rngs = {s: np.random.default_rng(ss) for s, ss in zip(MASKED_SPLITS, split_ss)}
    rng_w = np.random.default_rng(w_ss)
    d = len(groups)
    p = observed_probabilities(d, pattern, np.random.default_rng(plan_ss), p_col)
    missing = np.flatnonzero(p < 1)
    nested = pattern in RETURN
    if nested:
        missing = missing[np.argsort(-p[missing], kind="stable")]  # most observed first
    if mechanism == "mar":
        allowed = [j for j in range(d) if p[j] == 1]
        if not allowed:
            raise ValueError(f"MAR needs complete columns; pattern {pattern!r} has none")
    else:
        allowed = list(missing)
    feats = np.concatenate([groups[j] for j in allowed])

    masks = {s: np.ones((len(Z[s]), d), dtype=bool) for s in MASKED_SPLITS}
    reach = {s: np.ones(len(Z[s])) for s in MASKED_SPLITS}  # nested: P(observed in the previous column)
    prev = {s: np.ones(len(Z[s]), dtype=bool) for s in MASKED_SPLITS}
    for j in missing:
        if mechanism == "mcar":
            logit = {s: np.zeros(len(Z[s])) for s in MASKED_SPLITS}
        else:
            w = rng_w.standard_normal(len(feats))
            logit = {s: Z[s][:, feats] @ w for s in MASKED_SPLITS}
            sd = logit["train"].std()
            if sd > 0:
                logit = {s: v / sd for s, v in logit.items()}
        base = {s: reach[s] + RETURN[pattern] * (1 - reach[s]) for s in MASKED_SPLITS} if nested else reach
        b = brentq(lambda b: (base["train"] * expit(logit["train"] + b)).mean() - p[j], -50.0, 50.0, xtol=1e-12)
        for s in MASKED_SPLITS:
            h = expit(logit[s] + b)
            if nested:
                masks[s][:, j] = rngs[s].random(len(Z[s])) < h * np.where(prev[s], 1.0, RETURN[pattern])
                prev[s], reach[s] = masks[s][:, j], base[s] * h
            else:
                masks[s][:, j] = rngs[s].random(len(Z[s])) < h
    if (masks["train"].sum(0) < 2).any():
        raise ValueError("a column has fewer than 2 observed train entries")
    return masks
