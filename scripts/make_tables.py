"""README tables from runs/<dataset>/<pattern>-<mechanism>/seed<s>/<method>.json (validation split, x100).

    python scripts/make_tables.py --runs runs

Main table: every metric averaged over all runs (datasets x patterns x seeds). Utility rank: the methods are ranked
by Utility within every run (1 = best) and the ranks averaged. Per-dataset tables (Imp. Acc per pattern) follow
inside <details>.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pmae.run import METHODS

NAMES = {"pmae-mix": "PMAE-mix (ours)", "pmae-trf": "PMAE-trf (ours)", "remasker": "ReMasker",
         "hyperimpute": "HyperImpute", "miceforest": "MiceForest", "gain": "GAIN", "em": "EM", "knn": "KNN",
         "naive": "Naive"}  # row order of the tables
DATASETS = ("diabetes", "shoppers", "adult", "letter")  # order of the per-dataset tables
PATTERNS = {"monotone0.6-mnar": "Monotone", "quasi-mnar": "Quasi-Monotone", "general-mnar": "General",
            "partial-a0.6-mnar": "Partial-A", "partial-b-mnar": "Partial-B"}
METRICS = {"Imp. Acc ↑": "imp_acc", "R² ↑": "r2", "Acc ↑": "acc", "RMSE ↓": "rmse"}


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for f in runs.glob("*/*/seed*/*.json"):
        if f.stem not in METHODS:
            continue
        r = json.loads(f.read_text())
        val = r["imputation"]["val"]
        rows.append({"dataset": r["dataset"], "pattern": r["setting"], "seed": r["seed"], "method": f.stem,
                     **{k: val[v] * 100 for k, v in METRICS.items()}, "Utility ↑": r["utility"]["relative"] * 100})
    df = pd.DataFrame(rows)
    df["Utility (rank ↓)"] = df.groupby(["dataset", "pattern", "seed"])["Utility ↑"].rank(ascending=False)
    return df


def _bold_best(t: pd.DataFrame, lower_better=()) -> pd.DataFrame:
    out = t.map(lambda v: f"{v:.1f}")
    for c in t.columns:
        best = t[c].min() if c in lower_better else t[c].max()
        out.loc[t[c] == best, c] = f"**{best:.1f}**"
    return out


def markdown(t: pd.DataFrame) -> str:
    lines = ["| Method | " + " | ".join(t.columns) + " |", "|---|" + "---:|" * len(t.columns)]
    lines += [f"| {NAMES[m]} | " + " | ".join(row) + " |" for m, row in t.iterrows()]
    return "\n".join(lines)


def main_table(df: pd.DataFrame) -> str:
    cols = ["Imp. Acc ↑", "R² ↑", "Acc ↑", "RMSE ↓", "Utility (rank ↓)"]
    t = df.groupby("method")[cols].mean()
    t = t.reindex([m for m in NAMES if m in t.index])
    return markdown(_bold_best(t, lower_better=("RMSE ↓", "Utility (rank ↓)")))


def dataset_tables(df: pd.DataFrame) -> str:
    parts = []
    order = [d for d in DATASETS if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASETS))
    for dataset in order:
        t = df[df.dataset == dataset].pivot_table(index="method", columns="pattern", values="Imp. Acc ↑")
        t = t[[p for p in PATTERNS if p in t.columns]]
        t = t.rename(columns=PATTERNS)
        t["Mean"] = t.mean(axis=1)
        t = t.reindex([m for m in NAMES if m in t.index])
        parts.append(f"**{dataset.capitalize()}**, Imp. Acc ↑\n\n" + markdown(_bold_best(t)))
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path)
    args = ap.parse_args()
    df = load(args.runs)
    n = df.groupby("method").size()
    print(f"<!-- {len(df)} runs; per method: {n.min()}-{n.max()} -->\n")
    print(main_table(df), "\n")
    print("<details>\n<summary>Per dataset and pattern</summary>\n")
    print(dataset_tables(df))
    print("\n</details>")


if __name__ == "__main__":
    main()
