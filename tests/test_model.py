import numpy as np
import pytest
import torch

from pmae.baselines.remasker import ReMasker
from pmae.imputer import PMAE, batch_size_for, learning_rate
from pmae.masking import mask_rate, sample_visible, visible_rate
from pmae.model import CategoricalColumns, MaskedAutoencoder, categorical_loss, observed_loss


def _mixed_table(n=400, seed=0):
    """Two numerical columns and two categorical ones (3 and 2 categories), 30% missing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[:, 2] = rng.integers(0, 3, n)
    X[:, 3] = rng.integers(0, 2, n)
    X[rng.random(X.shape) < 0.3] = np.nan
    return X


def test_mask_rate_is_eq_11():
    p = torch.tensor([0.2, 0.5, 0.9])
    expected = 0.05 * torch.log((1 - p) / p) + 0.5
    assert torch.allclose(mask_rate(p), expected, atol=1e-6)
    assert torch.allclose(visible_rate(p) + mask_rate(p), torch.ones(3))
    assert visible_rate(torch.tensor([0.0]))[0] == 0  # clipped, no NaN


def test_sample_visible_hides_only_observed_entries_of_partial_columns():
    torch.manual_seed(0)
    n = 20000
    m = torch.ones(n, 3)
    m[: n // 5, 1] = 0  # column 1 observed in 80% of rows
    m[: n // 2, 2] = 0  # column 2 observed in 50% of rows
    v = sample_visible(m)
    assert torch.equal(v[:, 0], m[:, 0])  # fully observed: never hidden
    assert (v <= m).all()  # missing entries stay hidden
    for j, p in ((1, 0.8), (2, 0.5)):
        hidden = 1 - v[m[:, j] == 1, j].mean()
        assert abs(hidden - mask_rate(torch.tensor(p))) < 0.02


@pytest.mark.parametrize("blocks", ["mixer", "transformer"])
@pytest.mark.parametrize("categorical", [False, True])
def test_forward_shapes(blocks, categorical):
    torch.manual_seed(0)
    cat = CategoricalColumns([3, 4], [3, 2], [np.ones(3) / 3, np.ones(2) / 2]) if categorical else None
    model = MaskedAutoencoder(5, blocks=blocks, depth=2, decoder_depth=1, categorical=cat)
    m = (torch.rand(8, 5) > 0.3).float()
    x = torch.rand(8, 5)
    x[:, 3], x[:, 4] = torch.randint(0, 3, (8,)), torch.randint(0, 2, (8,))
    loss, pred = model(x * m, m)
    assert pred.shape == (8, 5) and torch.isfinite(loss)
    model.eval()
    loss, pred = model(x * m, m)
    assert loss is None
    if categorical:
        assert set(pred[:, 3].tolist()) <= {0.0, 1.0, 2.0} and set(pred[:, 4].tolist()) <= {0.0, 1.0}


def test_loss_is_eq_4():
    torch.manual_seed(0)
    pred, x = torch.rand(6, 3), torch.rand(6, 3)
    m = torch.tensor([[1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1.]])
    e = (pred - x) ** 2
    eq4 = sum((e[:, j] * m[:, j]).sum() / m[:, j].sum() for j in range(3))
    assert torch.allclose(observed_loss(pred, x, m), eq4)
    m[:, 2] = 0  # a column with no observed entry in the batch contributes nothing
    assert torch.isfinite(observed_loss(pred, x, m))


def test_uninformed_prediction_scores_one_per_column():
    """CDTD calibration: predicting the mean of a standardized column, or the category proportions, scores 1."""
    rng = np.random.default_rng(0)
    n = 20000
    z = torch.as_tensor(rng.normal(size=(n, 2)), dtype=torch.float32)
    z = (z - z.mean(0)) / z.std(0, unbiased=False)
    props = [np.array([0.5, 0.3, 0.2]), np.array([0.9, 0.1])]
    codes = torch.stack([torch.as_tensor(rng.choice(len(p), n, p=p)) for p in props], dim=1)
    empirical = [np.bincount(codes[:, k].numpy()) / n for k in range(2)]
    cat = CategoricalColumns([2, 3], [3, 2], empirical)
    logits = [torch.log(torch.as_tensor(p, dtype=torch.float32)).expand(n, -1) for p in empirical]
    m = torch.ones(n, 2)
    total = categorical_loss(logits, codes, m, cat.entropy, observed_loss(torch.zeros(n, 2), z, m))
    assert abs(float(total) - 4.0) < 1e-3


def test_category_embeddings_have_unit_scale_and_hidden_entries_read_zero():
    X = _mixed_table()
    imp = PMAE(epochs=1, device="cpu").fit(X, [2, 3])
    model = imp.model_.eval()
    Xs, M = imp._prepare(X)
    tokens = model._category_tokens(Xs, M)
    visible = M[:, model.cat_index].bool()
    assert torch.allclose(tokens[visible].norm(dim=1), torch.full((int(visible.sum()),), 32 ** 0.5), atol=1e-4)
    assert (tokens[~visible] == 0).all()


@pytest.mark.parametrize("cls", [PMAE, ReMasker])
def test_scaling_of_categorical_codes(cls):
    """PMAE reads category codes as they are; ReMasker min-max scales them like numerical values."""
    X = _mixed_table()
    imp = cls(epochs=1, device="cpu").fit(X, [2, 3])
    raw = (imp.scaler_.shift_[2:] == 0).all() and (imp.scaler_.scale_[2:] == 1).all()
    assert raw == (cls is PMAE)


@pytest.mark.parametrize("blocks", ["mixer", "transformer"])
def test_pmae_imputes_valid_codes_and_is_deterministic(blocks):
    X = _mixed_table()
    a = PMAE(blocks=blocks, epochs=3, device="cpu", seed=1).fit_transform(X, [2, 3])
    b = PMAE(blocks=blocks, epochs=3, device="cpu", seed=1).fit_transform(X, [2, 3])
    observed = ~np.isnan(X)
    assert not np.isnan(a).any() and np.array_equal(a[observed], X[observed]) and np.array_equal(a, b)
    assert set(np.unique(a[:, 2])) <= {0.0, 1.0, 2.0} and set(np.unique(a[:, 3])) <= {0.0, 1.0}


def test_fit_rejects_a_column_with_no_observed_entry():
    X = _mixed_table()
    X[:, 1] = np.nan
    with pytest.raises(ValueError, match="no observed entries"):
        PMAE(epochs=1, device="cpu").fit(X, [2, 3])


def test_batch_size_and_learning_rate_rules():
    assert [batch_size_for(n) for n in (999, 1000, 4999, 9999, 19999, 20000)] == [128, 256, 512, 1024, 2048, 4096]
    assert learning_rate("fixed", 1e-3, 1024) == 1e-3
    assert learning_rate("remasker", 1e-3, 128) == 2e-3
    assert PMAE().lr == "remasker" and ReMasker().lr == "fixed"
