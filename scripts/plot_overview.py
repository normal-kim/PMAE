"""README figure: which observed entries ReMasker and PMAE mask during training.

    python scripts/plot_overview.py    # writes assets/overview_{light,dark}.png

The example is a batch with monotone (block) missingness: columns are observed in 100%, 99%, ..., 2% of the rows,
and a row observed in a column is also observed in every column to its left. The masks follow the
training code's rules.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pmae.masking import mask_rate, sample_visible

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", line="#d8d7d2",
                  visible="#2a78d6", hidden="#eb6834", missing="#e6e5e0"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", line="#3a3a38",
                 visible="#3987e5", hidden="#d95926", missing="#2e2e2c"),
}
PROPORTIONS = [1.0, 1.0, 0.99, 0.98, 0.96, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05, 0.02]


def example_mask(n_rows: int = 100) -> torch.Tensor:
    """Monotone missingness with the observed proportions above; rows from most to least observed."""
    u = (torch.arange(n_rows) + 0.5) / n_rows
    return (u[:, None] < torch.tensor(PROPORTIONS)[None, :]).float()


def remasker_visible(m: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """ReMasker's training mask: every row keeps floor(d * (1 - ratio)) tokens, observed entries first in a random
    order. Observed entries are masked only in rows with more observed entries than that, in any column."""
    n, d = m.shape
    noise = torch.rand(n, d)
    noise[m < 1e-6] = 1
    keep = torch.zeros(n, d)
    keep.scatter_(1, torch.argsort(noise, dim=1)[:, :int(d * (1 - mask_ratio))], 1.0)
    return keep * m


def draw(m, masks, t, path):
    plt.rcParams.update({"font.size": 10, "text.color": t["text"], "axes.labelcolor": t["muted"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"]})
    fig = plt.figure(figsize=(11.5, 4.1), facecolor=t["surface"])
    cmap = ListedColormap([t["missing"], t["hidden"], t["visible"]])
    states = {"Observed Data": m * 2, "ReMasker (ICLR 2024)": m + masks["ReMasker"], "PMAE (ours)": m + masks["PMAE"]}
    for i, (name, grid) in enumerate(states.items()):
        ax = fig.add_axes([0.035 + i * 0.2, 0.17, 0.175, 0.73], facecolor=t["surface"])
        ax.imshow(grid.numpy(), cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(name, color=t["text"], fontsize=10.5, weight="bold", pad=6)
        for spine in ax.spines.values():
            spine.set_color(t["line"])
    legend = [Patch(color=t["visible"], label="observed"),
              Patch(color=t["hidden"], label="masked (to predict)"),
              Patch(color=t["missing"], label="missing")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.3225, 0.02), ncol=3, frameon=False,
               labelcolor=t["text"], fontsize=11, handlelength=1.4, handleheight=1.0, columnspacing=2.2)

    ax = fig.add_axes([0.69, 0.17, 0.285, 0.73], facecolor=t["surface"])
    p = torch.linspace(0.0005, 0.9995, 600)
    ax.plot(p, mask_rate(p), color=t["hidden"], lw=2.2, label="Proportional (PMAE)")
    ax.plot([0, 1], [0.5, 0.5], color=t["muted"], lw=2.2, label="Uniform (ReMasker)")
    ax.scatter([1.0], [0.0], s=36, color=t["hidden"], edgecolors=t["surface"], linewidths=0.6, zorder=3, clip_on=False)
    ax.annotate("fully observed:\nnot masked", xy=(1.0, 0.0), xytext=(0.8, 0.13), color=t["text"], fontsize=8.5,
                ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1))
    ax.set_xlim(0, 1.02), ax.set_ylim(0, 1)
    ax.set_xlabel(r"$p_{\mathrm{obs},j}$")
    ax.set_ylabel(r"$M_j(\cdot)$")
    ax.set_title("Masking function", color=t["text"], fontsize=10.5, weight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_color(t["line"])
    ax.grid(color=t["line"], lw=0.6)
    ax.legend(loc="upper right", frameon=False, labelcolor=t["text"], fontsize=9)
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main(n_rows: int = 100, out: Path | None = None, suffix: str = ""):
    torch.manual_seed(0)
    m = example_mask(n_rows)
    rules = {"ReMasker": remasker_visible, "PMAE": sample_visible}
    masks = {name: rule(m) for name, rule in rules.items()}
    out = out or Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        draw(m, masks, theme, out / f"overview_{name}{suffix}.png")


if __name__ == "__main__":
    main()
