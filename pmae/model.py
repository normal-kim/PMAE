"""The masked autoencoder of PMAE.

Tokens: one of width w per column, plus a [CLS] token, with a fixed sin-cos position embedding.
- Numerical column: its standardized value, through a linear map shared by all columns.
- Categorical column: a learned embedding of its category, L2-normalized to norm sqrt(w), the scale of a
  standardized value (as in CDTD, Mueller et al., ICLR 2025).
Masked and missing entries are read as 0 (the zero vector for categorical columns). The encoder and the decoder
stack MLP-mixer or transformer blocks; the decoder puts a learned mask token at the masked and missing positions.
Heads: a linear head shared by the numerical columns, and a softmax head per categorical column.

Loss (Eq. 4, calibrated as in CDTD): for each column, the loss of its observed entries, masked or not, averaged over
those entries in the batch, then summed over the columns. Numerical columns use the squared error, categorical
columns the cross-entropy divided by the entropy of their observed categories, so that an uninformed prediction
(the mean, or the category proportions) scores 1 on every column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .masking import sample_visible

LN_EPS = 1e-8


def sincos_pos_embed(width: int, n_tokens: int) -> torch.Tensor:
    """(1, n_tokens + 1, width) fixed 1-D sin-cos embedding; the first row (the [CLS] token) is zero."""
    omega = 1.0 / 10000 ** (np.arange(width // 2, dtype=float) / (width / 2.0))
    out = np.einsum("m,d->md", np.arange(n_tokens), omega)
    emb = np.concatenate([np.zeros([1, width]), np.concatenate([np.sin(out), np.cos(out)], axis=1)], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MixerBlock(nn.Module):
    """Token mixing across the d + 1 tokens, then channel mixing; dropout 0.1 in both MLPs."""

    def __init__(self, dim: int, n_tokens: int, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.norm_c = nn.LayerNorm(dim)
        self.norm_d = nn.LayerNorm(n_tokens)
        self.mlp_c = Mlp(dim, int(dim * mlp_ratio), drop)
        self.mlp_d = Mlp(n_tokens, int(n_tokens * mlp_ratio), drop)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = (x + self.mlp_d(self.norm_d(x))).transpose(1, 2)
        return x + self.mlp_c(self.norm_c(x))


def init_mae(model: nn.Module) -> None:
    """Initialization shared with ReMasker: xavier for the tokenizer and linear layers, N(0, 0.02) for the tokens."""
    w = model.embed.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.normal_(model.cls_token, std=0.02)
    nn.init.normal_(model.mask_token, std=0.02)

    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(_init)


def observed_loss(pred: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Eq. 4: sum_j sum_i m_ij (pred_ij - x_ij)^2 / sum_i m_ij over a batch with observed mask m."""
    return ((((pred - x) ** 2) * m).sum(0) / m.sum(0).clamp(min=1)).sum()


def categorical_loss(logits: list[torch.Tensor], codes: torch.Tensor, m: torch.Tensor, entropy: list[float],
                     loss: torch.Tensor | float = 0.0) -> torch.Tensor | float:
    """``loss`` plus, for every categorical column k, the cross-entropy averaged over its observed entries and
    divided by the entropy of the column's observed categories (columns with a single category are skipped)."""
    for k, lg in enumerate(logits):
        if entropy[k] > 0:
            ce = F.cross_entropy(lg.float(), codes[:, k], reduction="none")
            loss = loss + (ce * m[:, k]).sum() / m[:, k].sum().clamp(min=1) / entropy[k]
    return loss


@dataclass
class CategoricalColumns:
    """Columns that hold integer codes 0 .. K_j - 1: their positions, number of categories, and category
    proportions among the observed train entries."""

    index: list[int]
    n_categories: list[int]
    proportions: list[np.ndarray]

    @property
    def entropy(self) -> list[float]:
        return [float(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in self.proportions]


class MaskedAutoencoder(nn.Module):
    def __init__(self, d: int, width: int = 32, depth: int = 6, decoder_depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 4.0, blocks: str = "mixer", a: float = 0.05, b: float = 0.5,
                 categorical: CategoricalColumns | None = None):
        super().__init__()
        if blocks not in ("mixer", "transformer"):
            raise ValueError(f"blocks must be 'mixer' or 'transformer', got {blocks!r}")
        self.a, self.b, self.width = a, b, width

        def block():
            if blocks == "mixer":
                return MixerBlock(width, d + 1, mlp_ratio)
            return TransformerBlock(width, heads, mlp_ratio)

        self.embed = nn.Conv1d(1, width, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("pos_embed", sincos_pos_embed(width, d))
        self.blocks = nn.ModuleList([block() for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=LN_EPS)
        self.decoder_embed = nn.Linear(width, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("decoder_pos_embed", sincos_pos_embed(width, d))
        self.decoder_blocks = nn.ModuleList([block() for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(width, eps=LN_EPS)
        self.decoder_pred = nn.Linear(width, 1)
        init_mae(self)

        self.cat = categorical if categorical is not None and len(categorical.index) else None
        self.cat_heads = nn.ModuleList()
        if self.cat is None:
            return
        K = list(self.cat.n_categories)
        self.cat_entropy = self.cat.entropy
        self.register_buffer("cat_index", torch.tensor(self.cat.index, dtype=torch.long))
        self.register_buffer("num_index", torch.tensor([j for j in range(d) if j not in set(self.cat.index)],
                                                       dtype=torch.long))
        self.register_buffer("cat_max", torch.tensor(K) - 1)
        self.register_buffer("cat_offsets", torch.tensor(np.concatenate([[0], np.cumsum(K)[:-1]]), dtype=torch.long))
        self.cat_embed = nn.Embedding(sum(K), width)
        nn.init.normal_(self.cat_embed.weight)
        self.cat_heads = nn.ModuleList([nn.Linear(width, k) for k in K])
        for head in self.cat_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _codes(self, x):
        return torch.minimum(x[:, self.cat_index].long().clamp(min=0), self.cat_max)

    def _category_tokens(self, x, v):
        """(B, n_categorical, w): embeddings of the visible categories, zero elsewhere."""
        table = F.normalize(self.cat_embed.weight, dim=1) * self.width ** 0.5
        tokens = table[self._codes(x) + self.cat_offsets]
        return tokens * v[:, self.cat_index].unsqueeze(2).to(tokens.dtype)

    def encode(self, x, v):
        t = self.embed((x * v).unsqueeze(1)).transpose(1, 2)
        if self.cat is not None:
            t = t.index_copy(1, self.cat_index, self._category_tokens(x, v).to(t.dtype))
        t = t + self.pos_embed[:, 1:]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(t.shape[0], -1, -1)
        t = torch.cat([cls, t], dim=1)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)

    def decode(self, t, v):
        t = self.decoder_embed(t)
        t = torch.cat([t[:, :1], t[:, 1:] * v.unsqueeze(2) + (1 - v).unsqueeze(2) * self.mask_token], dim=1)
        t = t + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            t = blk(t)
        h = self.decoder_norm(t)
        logits = [head(h[:, 1 + j]) for head, j in zip(self.cat_heads, self.cat.index)] if self.cat else []
        return self.decoder_pred(h)[:, 1:, 0], logits

    def forward(self, x, m):
        """``x``: standardized numerical values and integer category codes, 0 at missing entries; ``m``: observed
        mask (1 = observed). Training returns (loss, prediction); evaluation returns (None, prediction) from all
        observed entries, with the predicted category codes in the categorical columns."""
        v = sample_visible(m, self.a, self.b) if self.training else m
        pred, logits = self.decode(self.encode(x, v), v)
        if self.cat is None:
            return (observed_loss(pred, x, m) if self.training else None), pred
        if not self.training:
            classes = torch.stack([lg.argmax(1) for lg in logits], dim=1).float()
            return None, pred.float().index_copy(1, self.cat_index, classes)
        n = self.num_index
        loss = observed_loss(pred[:, n], x[:, n], m[:, n]) if len(n) else 0.0
        return categorical_loss(logits, self._codes(x), m[:, self.cat_index], self.cat_entropy, loss), pred
