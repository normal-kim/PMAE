"""ReMasker (Du, Melis and Wang, ICLR 2024), following the official implementation.

Differences from PMAE: every row keeps half of its tokens, observed entries first, so observed entries are masked
only in rows that are more than half observed, fully observed columns included; the encoder reads only the kept
tokens; the loss averages the masked observed entries and the kept tokens separately over the whole batch.
"""
from __future__ import annotations

import torch
from torch import nn

from ..imputer import MAEImputer
from ..model import LN_EPS, TransformerBlock, init_mae, sincos_pos_embed


class ReMaskerModel(nn.Module):
    def __init__(self, d: int, width: int = 32, depth: int = 6, decoder_depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 4.0, mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed = nn.Conv1d(1, width, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("pos_embed", sincos_pos_embed(width, d))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=LN_EPS)
        self.decoder_embed = nn.Linear(width, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("decoder_pos_embed", sincos_pos_embed(width, d))
        self.decoder_blocks = nn.ModuleList([TransformerBlock(width, heads, mlp_ratio) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(width, eps=LN_EPS)
        self.decoder_pred = nn.Linear(width, 1)
        init_mae(self)

    def forward(self, x, m):
        N, L = x.shape
        t = self.embed(x.unsqueeze(1)).transpose(1, 2) + self.pos_embed[:, 1:]
        # keep a fixed number of tokens per row, observed entries first in a random order
        n_keep = int(L * (1 - self.mask_ratio)) if self.training else int(m.sum(1).min())
        noise = torch.rand(N, L, device=x.device)
        noise[m < 1e-6] = 1
        order = torch.argsort(noise, dim=1)
        restore = torch.argsort(order, dim=1)
        t = torch.gather(t, 1, order[:, :n_keep].unsqueeze(-1).repeat(1, 1, t.shape[2]))
        hidden = torch.ones(N, L, device=x.device)
        hidden[:, :n_keep] = 0
        hidden = torch.gather(hidden, 1, restore)
        visible = 1 - hidden
        if self.training:
            hidden[m < 1e-6] = 0

        cls = (self.cls_token + self.pos_embed[:, :1]).expand(N, -1, -1)
        t = torch.cat([cls, t], dim=1)
        for blk in self.blocks:
            t = blk(t)
        t = self.decoder_embed(self.norm(t))

        tokens = torch.cat([t[:, 1:], self.mask_token.repeat(N, L + 1 - t.shape[1], 1)], dim=1)
        tokens = torch.gather(tokens, 1, restore.unsqueeze(-1).repeat(1, 1, t.shape[2]))
        t = torch.cat([t[:, :1], tokens], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            t = blk(t)
        pred = (torch.tanh(self.decoder_pred(self.decoder_norm(t))) / 2 + 0.5)[:, 1:, 0]
        if not self.training:
            return None, pred
        e = (pred - x) ** 2
        return (e * hidden).sum() / hidden.sum() + (e * visible).sum() / visible.sum(), pred


class ReMasker(MAEImputer):
    """ReMasker as in the official code: every column, category codes included, min-max scaled to [0, 1]. The
    learning rate is fixed at 1e-3, the official rate at its batch size of 64 (the batches here are larger)."""

    scaling = "minmax"
    scale_categorical = True

    def __init__(self, mask_ratio: float = 0.5, width: int = 32, depth: int = 6, decoder_depth: int = 4,
                 heads: int = 4, lr: str = "fixed", **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.arch = dict(mask_ratio=mask_ratio, width=width, depth=depth, decoder_depth=decoder_depth, heads=heads)

    def _build(self, d: int) -> nn.Module:
        return ReMaskerModel(d, **self.arch)

    @torch.no_grad()
    def _predict(self, Xs: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Rows are grouped by their number of observed entries, so every row keeps all of them."""
        counts = M.sum(1).long()
        out = torch.empty_like(Xs)
        for k in counts.unique():
            idx = torch.nonzero(counts == k).squeeze(1)
            for i in range(0, len(idx), 8192):
                rows = idx[i:i + 8192]
                out[rows] = self.model_(Xs[rows], M[rows])[1].float()
        return out
