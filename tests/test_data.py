import numpy as np
import pandas as pd
import pytest

from pmae.data.patterns import make_masks, propensity_features
from pmae.data.preprocess import OTHER, CatVocab, merge_threshold
from pmae.data.scenario import build_scenario
from pmae.data.schema import Schema
from pmae.data.split import split_indices
from pmae.evaluate import imputation_metrics, r2_score


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n),
                       "k": rng.choice(["x", "y", "z"], n), "y": rng.normal(size=n)})
    return df, Schema("toy", "regression", ("a", "b", "c"), ("k",), "y")


def _masks(pattern, mechanism="mnar", seed=0):
    df, schema = _toy()
    tables = {"train": df.iloc[:400], "val": df.iloc[400:]}
    Z, groups = propensity_features(tables, schema)
    return make_masks(Z, groups, pattern, mechanism, seed)


def test_split_is_stratified_and_disjoint():
    y = np.array([0] * 80 + [1] * 20)
    s = split_indices(100, y=y)
    assert [len(s[k]) for k in ("train", "val", "test")] == [60, 20, 20]
    assert len(np.unique(np.concatenate(list(s.values())))) == 100
    assert y[s["test"]].mean() == pytest.approx(0.2)


@pytest.mark.parametrize("mechanism", ["mnar", "mar", "mcar"])
def test_monotone_is_nested(mechanism):
    for m in _masks("monotone", mechanism).values():
        share = m.mean(0)
        order = np.argsort(-share)
        assert (share == 1).sum() == 4 - round(0.6 * 4)  # the other columns stay complete
        assert all((m[:, order[k]] <= m[:, order[k - 1]]).all() for k in range(1, 4))  # observed only if observed before
        assert np.allclose(share[order[2:]], [0.8, 0.2], atol=0.06)  # the missing-prone columns


def test_partial_a_keeps_columns_complete():
    m = _masks("partial-a")["train"]
    missing = 1 - m.mean(0)
    assert (missing == 0).sum() == 4 - round(0.6 * 4)
    assert np.allclose(missing[missing > 0], 0.5, atol=0.06)


@pytest.mark.parametrize("pattern", ["quasi", "general", "partial-b"])
def test_every_column_gets_missing_values(pattern):
    assert (_masks(pattern)["train"].mean(0) < 1).all()


def test_quasi_is_nested_with_a_few_exceptions():
    m = _masks("quasi")["train"]
    order = np.argsort(-m.mean(0))
    returns = np.diff(m[:, order].astype(int), axis=1) > 0  # observed right after a missing column
    assert 0 < returns.sum() / m.sum() < 0.1


def test_mar_needs_complete_columns():
    with pytest.raises(ValueError):
        _masks("general", "mar")


def test_masks_depend_only_on_the_seed():
    a, b, c = _masks("general", seed=3), _masks("general", seed=3), _masks("general", seed=4)
    assert all(np.array_equal(a[s], b[s]) for s in a)
    assert not np.array_equal(a["train"], c["train"])


def test_category_merge_uses_observed_counts():
    assert merge_threshold(2000) == 30 and merge_threshold(5000) == 50
    X = np.array([["a"] * 40 + ["b"] * 40 + ["c"] * 5]).T.astype(object)
    M = np.ones_like(X, dtype=bool)
    vocab = CatVocab().fit(X, M)
    assert vocab.vocab_[0] == ["a", "b", OTHER]
    assert vocab.codes(np.array([["c"], ["unseen"], ["b"]], dtype=object)).ravel().tolist() == [2, 2, 1]
    M[:20] = False  # hide 20 "a": 20 observed < 30, so "a" is merged as well
    assert CatVocab().fit(X, M).vocab_[0] == ["b", OTHER]


def test_encoder_never_sees_missing_values():
    scn = build_scenario("diabetes", "general", seed=0)
    train = scn.tables["train"].copy()
    hidden = ~scn.masks["train"][:, 0]
    train.loc[hidden, scn.schema.num_cols[0]] = 1e6
    scn.encoder.fit(train, scn.masks["train"])
    assert np.array_equal(scn.encoder.transform(train, scn.masks["train"]), scn.X["train"], equal_nan=True)


def test_r2_is_zero_when_the_true_values_are_all_equal():
    t = np.full(50, 0.3)
    assert r2_score(t, t) == 0.0 and r2_score(t, t + 1) == 0.0  # undefined, not perfect
    assert r2_score(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])) == 1.0


def test_metrics_of_true_and_mean_imputation():
    scn = build_scenario("diabetes", "general", seed=0)
    truth = scn.encoder.transform(scn.tables["train"], np.ones_like(scn.masks["train"]))
    oracle = imputation_metrics(scn, "train", scn.finalize("train", truth))
    # true values outside the observed train range are clipped by the encoder, so numerical R2 stays below 1
    assert oracle["acc"] == 1.0 and 0.8 < oracle["r2"] < 1.0
    filled = scn.finalize("train", np.where(np.isnan(scn.X["train"]), np.nanmean(scn.X["train"], 0), 0))
    assert imputation_metrics(scn, "train", filled)["r2"] == 0.0  # a constant fill never beats the mean
    table = scn.imputed_table("train", filled)
    assert list(table.columns) == list(scn.schema.columns) and not table.isna().any().any()
