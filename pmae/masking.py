"""Proportional masking (Eq. 11 of the paper).

Within a batch, each observed entry of column j is masked with probability

    M_j = a * log((1 - p_j) / p_j) + b,   clipped to [0, 1],

where p_j is the observed proportion of column j in the batch: the less a column is observed, the more it is
masked. Columns that are fully observed in the batch are never masked.
"""
from __future__ import annotations

import torch


def mask_rate(p_obs: torch.Tensor, a: float = 0.05, b: float = 0.5) -> torch.Tensor:
    """M_j: the probability that an observed entry of a partially observed column is masked."""
    return 1 - visible_rate(p_obs, a, b)


def visible_rate(p_obs: torch.Tensor, a: float = 0.05, b: float = 0.5) -> torch.Tensor:
    """1 - M_j = a * log(p_j / (1 - p_j)) + 1 - b, clipped to [0, 1]."""
    return (a * torch.log(p_obs / (1 - p_obs)) + (1 - b)).clamp(0, 1)


def sample_visible(m: torch.Tensor, a: float = 0.05, b: float = 0.5) -> torch.Tensor:
    """Visible mask m+ for a batch with observed mask ``m`` (N x d, 1 = observed)."""
    p = m.mean(0)
    partial = p < 1
    keep = visible_rate(p[partial], a, b)
    u = torch.rand(m.shape[0], int(partial.sum()), device=m.device)
    v = m.clone()
    v[:, partial] = m[:, partial] * (u < keep)
    return v
