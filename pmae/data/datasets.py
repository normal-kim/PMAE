"""Step 1: download and clean the nine benchmark tables.

Each loader returns the complete table (no NaN) and its Schema. Numerical
columns are float64; categorical columns and a classification target are str.
Native missing categories (Adult's "?") are kept as their own category.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Schema

DATASETS = ("diabetes", "wine", "obesity", "bike", "shoppers", "letter", "default", "news", "adult")

UCI = "https://archive.ics.uci.edu/static/public"
# archive URL and sha256 of the data file(s) used inside it (Diabetes ships with scikit-learn)
SOURCES = {
    "wine": (f"{UCI}/186/wine+quality.zip", {
        "winequality-red.csv": "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"}),
    "obesity": (f"{UCI}/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip", {
        "ObesityDataSet_raw_and_data_sinthetic.csv":
            "28b6991062855d3d15642e38b1aa9d1b32fb7abbb5abf5b096f1868cd706154b"}),
    "bike": (f"{UCI}/560/seoul+bike+sharing+demand.zip", {
        "SeoulBikeData.csv": "373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60"}),
    "shoppers": (f"{UCI}/468/online+shoppers+purchasing+intention+dataset.zip", {
        "online_shoppers_intention.csv": "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"}),
    "letter": (f"{UCI}/59/letter+recognition.zip", {
        "letter-recognition.data": "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2"}),
    "default": (f"{UCI}/350/default+of+credit+card+clients.zip", {
        "default of credit card clients.xls": "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"}),
    "news": (f"{UCI}/332/online+news+popularity.zip", {
        "OnlineNewsPopularity/OnlineNewsPopularity.csv":
            "b66d9088632308cc27fa35af847650d174a5a50503987c4e511de94a99d1c218"}),
    "adult": (f"{UCI}/2/adult.zip", {
        "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
        "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"}),
}


def data_root() -> Path:
    return Path(os.environ.get("PMAE_DATA", "datasets"))


def load_dataset(name: str, root: str | Path | None = None, verify: bool = True) -> tuple[pd.DataFrame, Schema]:
    """Complete table and schema of ``name``; downloads and caches on first use."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    root = Path(root) if root is not None else data_root()
    out = root / name
    if (out / "complete.csv").exists() and (out / "schema.json").exists():
        schema = Schema.from_json(out / "schema.json")
        str_cols = list(schema.cat_cols) + ([] if schema.is_regression else [schema.target])
        df = pd.read_csv(out / "complete.csv", dtype={c: str for c in str_cols}, keep_default_na=False)
        return _finalize(df, schema), schema
    raw = None if name == "diabetes" else _download(name, root / "raw", verify)
    df, schema = LOADERS[name](raw)
    df = _finalize(df, schema)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"complete.{os.getpid()}.tmp"  # atomic: write, then rename
    df.to_csv(tmp, index=False)
    os.replace(tmp, out / "complete.csv")
    schema.to_json(out / "schema.json")
    return df, schema


def _finalize(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    df = df[list(schema.columns)].reset_index(drop=True)
    for c in schema.num_cols:
        df[c] = df[c].astype(np.float64)
    for c in schema.cat_cols:
        df[c] = df[c].astype(str)
    df[schema.target] = df[schema.target].astype(np.float64 if schema.is_regression else str)
    if df.isna().any().any():
        raise ValueError(f"{schema.name}: complete table contains NaN")
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(name: str, raw_root: Path, verify: bool) -> Path:
    url, files = SOURCES[name]
    dest = raw_root / name
    if not all((dest / f).exists() for f in files):
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "archive.zip"
        with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
        _safe_extract(archive, dest)
        for nested in list(dest.rglob("*.zip")):
            if nested != archive:
                _safe_extract(nested, nested.parent)
    if verify:
        for f, digest in files.items():
            if _sha256(dest / f) != digest:
                raise ValueError(f"{name}: {f} does not match the expected checksum; the source may have changed")
    return dest


def _safe_extract(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in archive: {member}")
        z.extractall(root)


def _cat_str(s: pd.Series) -> pd.Series:
    """Categories as strings; integer-valued codes without a trailing '.0'."""
    if pd.api.types.is_bool_dtype(s):
        return s.map({True: "True", False: "False"})
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.int64).astype(str)
    return s.astype(str).str.strip()


def _strings(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = _cat_str(df[c])
    return df


def _diabetes(_):
    from sklearn.datasets import load_diabetes

    df = load_diabetes(scaled=False, as_frame=True).frame.rename(columns={"target": "progression"})
    num = ("age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
    return _strings(df, ["sex"]), Schema("diabetes", "regression", num, ("sex",), "progression")


def _wine(raw: Path):
    df = pd.read_csv(raw / "winequality-red.csv", sep=";")
    num = tuple(c for c in df.columns if c != "quality")
    return _strings(df, ["quality"]), Schema("wine", "multiclass", num, (), "quality")


def _obesity(raw: Path):
    df = pd.read_csv(raw / "ObesityDataSet_raw_and_data_sinthetic.csv")
    num = ("Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE")
    cat = ("Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS")
    return _strings(df, cat + ("NObeyesdad",)), Schema("obesity", "multiclass", num, cat, "NObeyesdad")


def _bike(raw: Path):
    names = ["date", "rented_bike_count", "hour", "temperature", "humidity", "wind_speed", "visibility",
             "dew_point", "solar_radiation", "rainfall", "snowfall", "seasons", "holiday", "functioning_day"]
    df = pd.read_csv(raw / "SeoulBikeData.csv", encoding="latin-1", header=0, names=names).drop(columns="date")
    num = ("rented_bike_count", "temperature", "humidity", "wind_speed", "visibility", "dew_point",
           "solar_radiation", "rainfall", "snowfall")
    cat = ("hour", "seasons", "holiday")
    return _strings(df, cat + ("functioning_day",)), Schema("bike", "binclass", num, cat, "functioning_day")


def _shoppers(raw: Path):
    df = pd.read_csv(raw / "online_shoppers_intention.csv")
    num = tuple(df.columns[:10])
    cat = ("Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend")
    return _strings(df, cat + ("Revenue",)), Schema("shoppers", "binclass", num, cat, "Revenue")


def _letter(raw: Path):
    names = ["letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar",
             "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
    df = pd.read_csv(raw / "letter-recognition.data", header=None, names=names)
    cat = tuple(names[1:])
    return _strings(df, names), Schema("letter", "multiclass", (), cat, "letter")


def _default(raw: Path):
    df = pd.read_excel(raw / "default of credit card clients.xls", sheet_name="Data", header=1).drop(columns="ID")
    target = "default payment next month"
    cat = ("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
    num = tuple(c for c in df.columns if c not in cat + (target,))
    return _strings(df, cat + (target,)), Schema("default", "binclass", num, cat, target)


def _news(raw: Path):
    df = pd.read_csv(raw / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    channels = [c for c in df.columns if c.startswith("data_channel_is_")]
    days = [c for c in df.columns if c.startswith("weekday_is_")]
    onehot = df[channels].to_numpy()
    df["data_channel"] = np.where(onehot.sum(1) == 0, "none",
                                  np.array([c.removeprefix("data_channel_is_") for c in channels])[onehot.argmax(1)])
    df["weekday"] = np.array([d.removeprefix("weekday_is_") for d in days])[df[days].to_numpy().argmax(1)]
    df = df.drop(columns=["url", "is_weekend", *channels, *days])
    num = tuple(c for c in df.columns if c not in ("shares", "data_channel", "weekday"))
    return df, Schema("news", "regression", num, ("data_channel", "weekday"), "shares")


def _adult(raw: Path):
    cols = ["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation",
            "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week",
            "native.country", "income"]
    read = dict(header=None, names=cols, skipinitialspace=True, keep_default_na=False)
    df = pd.concat([pd.read_csv(raw / "adult.data", **read),
                    pd.read_csv(raw / "adult.test", skiprows=1, **read)], ignore_index=True)
    df["income"] = df["income"].str.rstrip(".")
    num = ("age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week")
    cat = tuple(c for c in cols[:-1] if c not in num)
    return _strings(df, cat + ("income",)), Schema("adult", "binclass", num, cat, "income")


LOADERS = {"diabetes": _diabetes, "wine": _wine, "obesity": _obesity, "bike": _bike, "shoppers": _shoppers,
           "letter": _letter, "default": _default, "news": _news, "adult": _adult}
