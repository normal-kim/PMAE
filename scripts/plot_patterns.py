"""README figure: the observed masks of the five missing patterns.

    python scripts/plot_patterns.py    # writes assets/patterns_{light,dark}.png

Masks of 200 training rows of Shoppers (MNAR, seed 0), drawn by the pattern generator. Columns go from most to least
observed, and rows from most to least observed entries.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from plot_overview import THEMES

from pmae.data.patterns import PATTERNS
from pmae.data.scenario import build_scenario

TITLES = {"monotone": ("Monotone", "nested"), "quasi": ("Quasi-Monotone", "nested, few exceptions"),
          "general": ("General", "independent"), "partial-a": ("Partial-A", "independent, some columns"),
          "partial-b": ("Partial-B", "independent, uneven rates")}


def ordered(m: np.ndarray) -> np.ndarray:
    m = m[:, np.argsort(-m.mean(0), kind="stable")]
    rows = np.lexsort(m.T[::-1])[::-1]  # ties broken by the pattern itself, so blocks stay together
    return m[rows[np.argsort(-m[rows].sum(1), kind="stable")]]


def draw(masks: dict, t: dict, path: Path):
    plt.rcParams.update({"font.size": 10, "text.color": t["text"]})
    fig = plt.figure(figsize=(12, 3.5), facecolor=t["surface"])
    cmap = ListedColormap([t["missing"], t["visible"]])
    for i, (pattern, m) in enumerate(masks.items()):
        ax = fig.add_axes([0.012 + i * 0.198, 0.2, 0.178, 0.62], facecolor=t["surface"])
        ax.imshow(ordered(m), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        title, structure = TITLES[pattern]
        ax.set_title(title, color=t["text"], fontsize=10.5, weight="bold", pad=17)
        ax.text(0.5, 1.03, structure, transform=ax.transAxes, ha="center", va="bottom", color=t["muted"], fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(t["muted"]), spine.set_linewidth(0.8)
    legend = [Patch(facecolor=t["visible"], edgecolor=t["muted"], linewidth=0.6, label="observed"),
              Patch(facecolor=t["missing"], edgecolor=t["muted"], linewidth=0.6, label="missing")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False,
               labelcolor=t["text"], fontsize=10, handlelength=1.4, handleheight=1.0, columnspacing=2.2)
    fig.savefig(path, dpi=150, facecolor=t["surface"])
    plt.close(fig)


def main(dataset: str = "shoppers", n_rows: int = 200, out: Path | None = None):
    masks = {}
    for pattern in PATTERNS:
        m = build_scenario(dataset, pattern, seed=0).masks["train"]
        masks[pattern] = m[np.random.default_rng(0).choice(len(m), n_rows, replace=False)]
    out = out or Path(__file__).resolve().parents[1] / "assets"
    out.mkdir(exist_ok=True)
    for name, theme in THEMES.items():
        draw(masks, theme, out / f"patterns_{name}.png")


if __name__ == "__main__":
    main()
