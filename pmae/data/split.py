"""Step 2: a fixed train/val/test split of the complete table."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

SPLITS = ("train", "val", "test")


def split_indices(
    n_rows: int,
    y: np.ndarray | None = None,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Row indices of the train/val/test splits.

    Stratified on ``y`` when it is given (classification). Indices are sorted so
    every split keeps the row order of the complete table.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    train_frac, val_frac, test_frac = fractions
    idx = np.arange(n_rows)
    train_val, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
    train, val = train_test_split(
        train_val,
        test_size=val_frac / (train_frac + val_frac),
        random_state=seed,
        stratify=None if y is None else y[train_val],
    )
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
