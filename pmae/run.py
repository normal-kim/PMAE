"""Impute one scenario with one method and evaluate it.

    python -m pmae.run --dataset diabetes --pattern general --method pmae-mix --seed 0
    python -m pmae.run --dataset adult --pattern quasi --method remasker --seed 1

The imputer is fit on the masked train split and fills the train and validation splits. Results go to
<out>/<dataset>/<pattern>-<mechanism>/seed<seed>/<method>[@options].json, with the imputed train and
validation splits (model space) in <method>_imputed.npz.
From Python, ``run(..., imputer=...)`` evaluates any object with fit(X, categorical) and transform(X).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data.datasets import DATASETS
from .data.patterns import MECHANISMS, PATTERNS
from .data.scenario import build_scenario
from .evaluate import downstream_scores, imputation_metrics, utility

METHODS = ("pmae-mix", "pmae-trf", "remasker", "hyperimpute", "miceforest", "gain", "em", "knn", "naive")
OPTIONS = {"epochs": int, "a": float, "b": float}


def make_imputer(method: str, seed: int = 0, device: str | None = None, **options):
    if method in ("pmae-mix", "pmae-trf"):
        from .imputer import PMAE

        return PMAE(blocks="mixer" if method == "pmae-mix" else "transformer", seed=seed, device=device, **options)
    if method == "remasker":
        from .baselines.remasker import ReMasker

        return ReMasker(seed=seed, device=device, **options)
    if options:
        raise ValueError(f"{method} takes no options")
    if method == "naive":
        from .baselines.simple import Naive

        return Naive()
    if method == "knn":
        from .baselines.simple import KNN

        return KNN()
    if method == "miceforest":
        from .baselines.external import MiceForest

        return MiceForest(seed=seed)
    if method == "gain":
        from .baselines.external import GAIN

        return GAIN(seed=seed)
    if method == "em":
        from .baselines.external import EM

        return EM()
    if method == "hyperimpute":
        from .baselines.external import HyperImpute

        return HyperImpute(seed=seed)
    raise ValueError(f"method must be one of {METHODS}")


def run_name(method: str, options: dict) -> str:
    return method + ("@" + ",".join(f"{k}={options[k]}" for k in sorted(options)) if options else "")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # atomic: write, then rename
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def reference_scores(dataset: str, scn, out: str | Path) -> dict[str, float]:
    """Utility reference: the downstream scores of the complete train table (computed once per dataset)."""
    path = Path(out) / dataset / "reference.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tables = scn.tables
        _write_json(path, {"downstream": downstream_scores(tables["train"], tables["val"], tables["test"], scn.schema)})
    return json.loads(path.read_text())["downstream"]


def run(dataset: str, pattern: str, method: str, seed: int = 0, mechanism: str = "mnar", p_col: float = 0.6,
        out: str | Path = "runs", device: str | None = None, imputer=None, **options) -> dict:
    scn = build_scenario(dataset, pattern, mechanism, seed, p_col)
    imputer = imputer if imputer is not None else make_imputer(method, seed, device, **options)

    start = time.time()
    imputer.fit(scn.X["train"], scn.categorical)
    fit_seconds = time.time() - start
    Z = {s: scn.finalize(s, imputer.transform(scn.X[s])) for s in ("train", "val")}
    seconds = time.time() - start

    scores = downstream_scores(scn.imputed_table("train", Z["train"]), scn.imputed_table("val", Z["val"]),
                               scn.tables["test"], scn.schema)
    ref = reference_scores(dataset, scn, out)
    result = {
        "dataset": dataset, "setting": scn.name, "seed": seed, "method": method, "options": options,
        "imputation": {s: imputation_metrics(scn, s, Z[s]) for s in ("train", "val")},
        "utility": {"relative": utility(scores, ref, scn.schema), "scores": scores, "reference": ref},
        "seconds": {"fit": round(fit_seconds, 1), "fit_and_impute": round(seconds, 1)},
        "checks": {"nan": int(sum(np.isnan(z).sum() for z in Z.values())),
                   "rows": {s: int(len(z)) for s, z in Z.items()},
                   "skipped_steps": getattr(imputer, "skipped_steps_", None)},
    }
    run_dir = Path(out) / dataset / scn.name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(method, options)
    tmp = run_dir / f"{name}_imputed.{os.getpid()}.tmp.npz"  # model-space imputations, to re-evaluate without refitting
    np.savez_compressed(tmp, train=Z["train"].astype(np.float32), val=Z["val"].astype(np.float32))
    os.replace(tmp, run_dir / f"{name}_imputed.npz")
    _write_json(run_dir / f"{name}.json", result)
    if getattr(imputer, "history_", None):
        pd.DataFrame({"epoch": range(len(imputer.history_)), "loss": imputer.history_}).to_csv(
            run_dir / f"{name}_loss.csv", index=False)
    return result


def _parse_set(items) -> dict:
    options = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in OPTIONS:
            raise SystemExit(f"unknown option {key!r}; choose from {sorted(OPTIONS)}")
        options[key] = OPTIONS[key](value)
    return options


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--pattern", required=True, choices=PATTERNS)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mechanism", default="mnar", choices=MECHANISMS)
    ap.add_argument("--p_col", type=float, default=0.6, help="share of columns with missing values (monotone, partial-a)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help=f"model option: {', '.join(OPTIONS)}")
    args = ap.parse_args()
    options = _parse_set(args.set)
    result = run(args.dataset, args.pattern, args.method, args.seed, args.mechanism, args.p_col, args.out,
                 args.device, **options)
    print(json.dumps({k: result[k] for k in ("setting", "method", "options", "utility", "seconds")}
                     | {"imp_acc": {s: v["imp_acc"] for s, v in result["imputation"].items()}}, indent=2))


if __name__ == "__main__":
    main()
