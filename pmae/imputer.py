"""fit / transform imputers around a masked autoencoder, and the training loop shared with ReMasker.

Training follows ReMasker's setup (Appendix A of the paper): AdamW (betas 0.9, 0.95; default weight decay),
a linear warm-up of 40 epochs followed by a half-cosine decay to 1e-5, updated every iteration, 600 epochs,
mixed precision on GPU, and a batch size that grows with the number of rows. The final weights are kept.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import CategoricalColumns, MaskedAutoencoder

LR_RULES = ("fixed", "remasker")
SCALINGS = ("minmax", "standard")


def batch_size_for(n_rows: int) -> int:
    for limit, size in ((1000, 128), (2500, 256), (5000, 512), (10000, 1024), (20000, 2048)):
        if n_rows < limit:
            return size
    return 4096


def learning_rate(rule: str, base_lr: float, batch_size: int) -> float:
    """"fixed": ``base_lr`` at every batch size; "remasker": ``base_lr * batch_size / 64``."""
    if rule not in LR_RULES:
        raise ValueError(f"lr must be one of {LR_RULES}, got {rule!r}")
    return base_lr if rule == "fixed" else base_lr * batch_size / 64


def lr_at(epoch: float, lr: float, min_lr: float, epochs: int, warmup: int) -> float:
    if epoch < warmup:
        return lr * epoch / warmup
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))


def train(model, X: torch.Tensor, M: torch.Tensor, epochs: int, batch_size: int, lr: float,
          min_lr: float = 1e-5, warmup: int = 40) -> tuple[list[float], int]:
    """Train ``model`` (forward(x, m) -> (loss, prediction)) on shuffled mini-batches of X with observed mask M,
    skipping steps with a non-finite loss. Returns the root mean loss of each epoch and the number of skipped steps."""
    amp = X.device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))
    loader = DataLoader(TensorDataset(X, M), batch_size=batch_size, shuffle=True)
    history, skipped = [], 0
    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for it, (xb, mb) in enumerate(loader):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(it / len(loader) + epoch, lr, min_lr, epochs, warmup)
            with torch.autocast("cuda", enabled=amp):
                loss, _ = model(xb, mb)
            value = loss.item()
            if not math.isfinite(value):
                optimizer.zero_grad()
                skipped += 1
                continue
            total, n = total + value, n + 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        history.append((total / max(n, 1)) ** 0.5)
    return history, skipped


class _Scaler:
    """Per-column scaling fit on the observed entries: "minmax" to [0, 1] (as in ReMasker) or "standard".
    The columns in ``unscaled`` pass through unchanged."""

    def __init__(self, kind: str, unscaled=()):
        if kind not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}, got {kind!r}")
        self.kind, self.unscaled = kind, list(unscaled)

    def fit(self, X: np.ndarray) -> "_Scaler":
        if self.kind == "minmax":
            self.shift_ = np.nanmin(X, 0).astype(np.float64)
            self.scale_ = np.nanmax(X, 0).astype(np.float64) - self.shift_ + 1e-8
        else:
            self.shift_ = np.nanmean(X, 0).astype(np.float64)
            std = np.nanstd(X, 0).astype(np.float64)
            self.scale_ = np.where(std > 0, std, 1.0)
        self.shift_[self.unscaled], self.scale_[self.unscaled] = 0.0, 1.0
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - torch.from_numpy(self.shift_).float()) / torch.from_numpy(self.scale_).float()

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        return Z * torch.from_numpy(self.scale_).float() + torch.from_numpy(self.shift_).float()


class MAEImputer:
    """Base class: scaling, training and imputation; subclasses set the scaling and build the model."""

    scaling = "standard"
    scale_categorical = False  # whether the categorical codes are scaled like numerical values

    def __init__(self, epochs: int = 600, lr: str = "remasker", base_lr: float = 1e-3,
                 batch_size: int | None = None, seed: int = 0, device: str | None = None):
        self.epochs, self.lr, self.base_lr = epochs, lr, base_lr
        self.batch_size, self.seed = batch_size, seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _build(self, d: int) -> torch.nn.Module:
        raise NotImplementedError

    def _prepare(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        M = (~torch.isnan(X)).float()
        return torch.nan_to_num(self.scaler_.transform(X)).to(self.device), M.to(self.device)

    def _rng(self):
        return torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else [])

    @staticmethod
    def _categories(X: np.ndarray, categorical: list[int]) -> CategoricalColumns:
        counts = [np.bincount(X[~np.isnan(X[:, j]), j].astype(np.int64)) for j in categorical]
        return CategoricalColumns(categorical, [len(c) for c in counts], [c / c.sum() for c in counts])

    def fit(self, X: np.ndarray, categorical=()) -> "MAEImputer":
        """``X``: (n, d) float array with NaN where missing; ``categorical``: positions of the columns that hold
        integer codes 0 .. K_j - 1."""
        X = np.asarray(X, dtype=np.float32)
        empty = np.flatnonzero(np.isnan(X).all(0))
        if len(empty):
            raise ValueError(f"columns {empty.tolist()} have no observed entries")
        categorical = list(categorical)
        self.scaler_ = _Scaler(self.scaling, () if self.scale_categorical else categorical).fit(X)
        self.categories_ = self._categories(X, categorical)
        Xs, M = self._prepare(X)
        batch_size = self.batch_size or batch_size_for(len(X))
        with self._rng():
            torch.manual_seed(self.seed)
            self.model_ = self._build(X.shape[1]).to(self.device)
            self.history_, self.skipped_steps_ = train(self.model_, Xs, M, self.epochs, batch_size,
                                                       learning_rate(self.lr, self.base_lr, batch_size))
        return self

    @torch.no_grad()
    def _predict(self, Xs: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.model_(Xs[i:i + 8192], M[i:i + 8192])[1] for i in range(0, len(Xs), 8192)])

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xs, M = self._prepare(X)
        self.model_.eval()
        with self._rng():
            torch.manual_seed(self.seed)
            pred = self._predict(Xs, M)
        pred = self.scaler_.inverse(pred.float().cpu()).numpy().astype(np.float64)
        return np.where(np.isnan(X), pred, X)

    def fit_transform(self, X: np.ndarray, categorical=()) -> np.ndarray:
        return self.fit(X, categorical).transform(X)


class PMAE(MAEImputer):
    """PMAE: proportional masking with an MLP-mixer ("mixer", PMAE-mix) or transformer (PMAE-trf) backbone.

    Example::

        # X: (n, d) floats with NaN where missing; columns 3 and 4 hold integer category codes 0 .. K - 1
        X_filled = PMAE(blocks="mixer").fit_transform(X, categorical=[3, 4])
    """

    def __init__(self, blocks: str = "mixer", a: float = 0.05, b: float = 0.5, width: int = 32, depth: int = 6,
                 decoder_depth: int = 4, heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.arch = dict(blocks=blocks, a=a, b=b, width=width, depth=depth, decoder_depth=decoder_depth, heads=heads)

    def _build(self, d: int) -> torch.nn.Module:
        return MaskedAutoencoder(d, categorical=self.categories_, **self.arch)
