# PMAE

Official code for **To Predict or Not to Predict? Proportionally Masked Autoencoders for Tabular Data Imputation** (AAAI 2025).
Jungkyu Kim, Kibok Lee, Taeyoung Park.

[![Paper](https://img.shields.io/badge/Paper-AAAI%202025-1f6feb)](https://ojs.aaai.org/index.php/AAAI/article/view/33967)
[![arXiv](https://img.shields.io/badge/arXiv-2412.19152-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2412.19152)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://github.com/normal-kim/PMAE/blob/main/LICENSE)

## 💡 Why PMAE

**Missingness differs from column to column, and the masking used to train an imputer should follow it.** A masked autoencoder for imputation masks part of the observed entries during training and learns to predict them from the remaining ones.

- **ReMasker** (ICLR 2024) adopts the naive uniform masking strategy on the observed entries, regardless of each column's missingness.
- **PMAE** masks the observed entries of column *j* with probability *M<sub>j</sub>* = *a* · log((1 − *p<sub>j</sub>*) / *p<sub>j</sub>*) + *b* (*a* = 0.05, *b* = 0.5), where *p<sub>j</sub>* is the column's observed proportion in the batch, leaving fully observed columns unmasked.

**Idea.** Following the principle of inverse probability weighting ([Rosenbaum & Rubin, 1983](https://doi.org/10.1093/biomet/70.1.41); [Seaman & White, 2013](https://doi.org/10.1177/0962280210395740)), less frequently observed variables should carry greater weight. In other words, the less a column is observed, the larger the fraction of its observed entries that become *prediction targets*.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/normal-kim/PMAE/main/assets/overview_dark.png">
  <img src="https://raw.githubusercontent.com/normal-kim/PMAE/main/assets/overview_light.png" alt="On a batch with monotone missingness, ReMasker masks observed entries only in the rows that are more than half observed, fully observed columns included, while PMAE masks every partially observed column at its rate M_j and never the fully observed ones; the right panel shows the two masking functions against the observed proportion." width="100%">
</picture>

## 🏗️ Architecture

```text
row with masked entries
  ─► tokens    numerical value → shared linear map, category → learned embedding; + position, [CLS]
  ─► encoder   6 blocks; masked and missing entries read as zero
  ─► decoder   4 blocks; a learned mask token at the masked and missing positions
  ─► heads     a value per numerical column, a softmax over the categories per categorical column
  ─► loss      per column, the mean error over its observed entries, masked or not (Eq. 4)
```

Both encoder and decoder stack pre-norm residual blocks of one of two types (Eqs. 12–13 of the paper):

| | Mixing across columns | Mixing within a column token |
|---|---|---|
| **PMAE-trf** | x ← x + SelfAttention(LN(x)), 4 heads | x ← x + MLP(LN(x)) |
| **PMAE-mix** | x ← x + MLP<sub>tokens</sub>(LN(x)), an MLP over the column axis | x ← x + MLP(LN(x)) |

Columns of tabular data often interact in groups rather than in pairs ([T2G-Former](https://ojs.aaai.org/index.php/AAAI/article/view/26272), AAAI 2023). Self-attention combines columns through soft pairwise weights, whereas the non-linear token-mixing MLP of [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NeurIPS 2021) can capture such group interactions more flexibly.

## ⚙️ Install

Python 3.10–3.12.

```bash
pip install pmae
```

To run the experiments, clone this repository and create the exact environment used for the results:

```bash
git clone https://github.com/normal-kim/PMAE.git && cd PMAE
conda env create -f environment.yml && conda activate pmae
```

## 🚀 Use PMAE on your own data

```python
from pmae import PMAE

imputer = PMAE(blocks="mixer")                           # blocks="transformer" for PMAE-trf
X_filled = imputer.fit_transform(X, categorical=[3, 4])
X_new_filled = imputer.transform(X_new)                  # rows the imputer was not fit on
```

| Input | Shape | Type | Content |
|---|---|---|---|
| `X` | `(n, d)` | float | numerical columns as values, categorical columns as integer codes `0 … K_j−1`; `NaN` where missing |
| `categorical` | `d_cat` | list of int | positions of the categorical columns |

It returns `X` with every missing entry filled: numerical values on the input scale, and a category code for every categorical entry.

### Build your own imputer

Any object with `fit(X, categorical)` and `transform(X)` can be evaluated on the same scenarios. It receives the encoded splits (step 4 of the Pipeline below): quantile-normal numerical columns and integer category codes, with `NaN` where missing.

```python
from pmae.run import run

class MyImputer:
    def fit(self, X, categorical):
        ...
        return self

    def transform(self, X):
        ...  # X with every NaN filled; integer codes in the categorical columns

run("adult", "general", "my-imputer", imputer=MyImputer())
```

## 🏃 Run

Datasets are downloaded from UCI on first use (Diabetes ships with scikit-learn): `diabetes`, `shoppers`, `adult` and `letter` (the results below), and `wine`, `obesity`, `bike`, `default` and `news` (also used in the paper).

```bash
# one method on one scenario: impute and evaluate  ->  runs/<dataset>/<pattern>-<mechanism>/seed<s>/<method>.json
python -m pmae.run --dataset adult --pattern general --method pmae-mix

# all experiments (4 datasets x 5 patterns x 3 seeds x 9 methods) followed by the summary tables
bash scripts/run_all.sh 0                  # on GPU 0
bash scripts/run_all.sh 1 adult letter     # or split the datasets across GPUs and servers,
python scripts/make_tables.py              # then summarize the merged runs/
```

Methods are `pmae-mix` and `pmae-trf` (ours), and the baselines `remasker`, `hyperimpute`, `miceforest`, `gain`, `em`, `knn` and `naive`. Missing values follow one of five patterns, with a logistic model of the complete values that depends on the missing columns themselves (MNAR):

| Pattern | Columns with missing values | Observed probability per column | Structure |
|---|---|---|---|
| `monotone` | 60% of the columns | decreasing evenly from 0.8 to 0.2 | nested: a row missing a column also misses every later one |
| `quasi` | all | 60% of the columns U(0.95, 0.99), the others U(0.2, 0.8) | nested, with a few exceptions |
| `general` | all | U(0.2, 0.8) | independent across columns |
| `partial-a` | 60% of the columns | 0.5 | independent across columns |
| `partial-b` | all | 60% of the columns U(0.95, 0.99), the others U(0.2, 0.8) | independent across columns |

Partial-A and Partial-B are the Monotone and Quasi-Monotone patterns of the paper's experiments. In this release, Monotone and Quasi-Monotone are updated to the nested structure of their definitions (see Notes on this release below).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/normal-kim/PMAE/main/assets/patterns_dark.png">
  <img src="https://raw.githubusercontent.com/normal-kim/PMAE/main/assets/patterns_light.png" alt="Observed masks of 200 Shoppers training rows under the five patterns: Monotone forms a clean staircase, Quasi-Monotone a staircase with a few scattered exceptions, General is scattered over every column, Partial-A leaves some columns complete and scatters the rest, and Partial-B is mostly observed with scattered gaps." width="100%">
</picture>

## 📏 Evaluation

Each imputer is fit on the train split. Imp. Acc and RMSE are computed on the held-out validation split, whose rows the imputer never sees during fitting, over the entries the pattern made missing.

- **Imp. Acc** (the paper's Imputation Accuracy): the mean, over the columns with missing entries, of R² for numerical columns (on a quantile-normal scale fit on the complete train split, clipped at 0) and accuracy for categorical columns.
- **RMSE** of the numerical entries on the same scale, min-max scaled to [0, 1].
- **Utility**: downstream performance of XGBoost, random forest and a linear model trained on the imputed data, relative to training on the complete data; reported as the rank of each method within a run.

These are **point accuracy** measures: they compare each imputed value with the true value, and do not measure whether the imputations follow the distribution of the data (their spread, or the dependence between columns).

## 📊 Results

Diabetes, Shoppers, Adult and Letter; the five patterns (MNAR); three seeds (×100, best in bold). Utility is the rank of each method within a run (1 = best), averaged over the runs.

Baselines: [ReMasker](https://proceedings.iclr.cc/paper_files/paper/2024/hash/258d234a229d1612a565f9d9a63e3e80-Abstract-Conference.html) (ICLR 2024; [code](https://github.com/tydusky/remasker)), [HyperImpute](https://proceedings.mlr.press/v162/jarrett22a.html) (ICML 2022; [code](https://github.com/vanderschaarlab/hyperimpute)), [MiceForest](https://github.com/AnotherSamWilson/miceforest) (chained equations with LightGBM), [GAIN](https://proceedings.mlr.press/v80/yoon18a.html) (ICML 2018), EM (Dempster et al., 1977), KNN (Troyanskaya et al., 2001) and mean / mode imputation (Naive).

| Method | Imp. Acc ↑ | R² ↑ | Acc ↑ | RMSE ↓ | Utility (rank ↓) |
|---|---:|---:|---:|---:|---:|
| PMAE-mix (ours) | **42.3** | **26.3** | **61.7** | **14.3** | **3.5** |
| PMAE-trf (ours) | 39.8 | 23.6 | 60.2 | 14.5 | 3.8 |
| ReMasker | 28.1 | 16.2 | 48.8 | 15.8 | 5.5 |
| HyperImpute | 29.8 | 16.3 | 47.9 | 16.1 | 5.4 |
| MiceForest | 30.6 | 16.7 | 51.9 | 15.7 | 4.9 |
| GAIN | 17.3 | 5.3 | 36.3 | 20.3 | 6.4 |
| EM | 28.0 | 16.9 | 44.0 | 15.9 | 5.5 |
| KNN | 27.9 | 10.6 | 49.6 | 17.3 | 4.7 |
| Naive | 18.0 | 0.0 | 42.2 | 19.0 | 5.3 |

<details>
<summary>Per dataset and pattern</summary>

**Diabetes**, Imp. Acc ↑

| Method | Monotone | Quasi-Monotone | General | Partial-A | Partial-B | Mean |
|---|---:|---:|---:|---:|---:|---:|
| PMAE-mix (ours) | 23.2 | 23.4 | **24.0** | 29.3 | 35.0 | 27.0 |
| PMAE-trf (ours) | 12.9 | 21.8 | 16.6 | 15.0 | 34.1 | 20.1 |
| ReMasker | 9.5 | 8.8 | 8.5 | 3.0 | 18.6 | 9.7 |
| HyperImpute | 12.2 | 16.7 | 12.7 | 18.8 | 24.8 | 17.0 |
| MiceForest | 13.3 | **24.0** | 11.8 | 22.2 | 26.3 | 19.5 |
| GAIN | 7.0 | 10.1 | 3.9 | 11.7 | 15.9 | 9.7 |
| EM | **25.0** | 23.2 | 21.6 | **30.5** | **36.5** | **27.4** |
| KNN | 17.8 | 16.2 | 5.8 | 16.2 | 25.2 | 16.2 |
| Naive | 0.0 | 1.4 | 4.2 | 0.0 | 5.1 | 2.1 |

**Shoppers**, Imp. Acc ↑

| Method | Monotone | Quasi-Monotone | General | Partial-A | Partial-B | Mean |
|---|---:|---:|---:|---:|---:|---:|
| PMAE-mix (ours) | **34.4** | **44.9** | **43.3** | **43.7** | **55.8** | **44.4** |
| PMAE-trf (ours) | 33.1 | 44.3 | 42.5 | 40.1 | 54.3 | 42.9 |
| ReMasker | 30.4 | 36.1 | 36.2 | 37.2 | 49.3 | 37.9 |
| HyperImpute | 28.0 | 36.2 | 29.7 | 29.1 | 46.1 | 33.8 |
| MiceForest | 25.2 | 35.1 | 29.4 | 32.6 | 47.0 | 33.9 |
| GAIN | 19.1 | 21.4 | 18.9 | 23.7 | 27.4 | 22.1 |
| EM | 27.7 | 33.5 | 32.6 | 29.5 | 41.4 | 33.0 |
| KNN | 29.0 | 19.7 | 18.1 | 28.9 | 38.8 | 26.9 |
| Naive | 17.7 | 21.5 | 22.5 | 17.5 | 22.7 | 20.4 |

**Adult**, Imp. Acc ↑

| Method | Monotone | Quasi-Monotone | General | Partial-A | Partial-B | Mean |
|---|---:|---:|---:|---:|---:|---:|
| PMAE-mix (ours) | **38.4** | **47.9** | **51.3** | **45.5** | 55.6 | **47.7** |
| PMAE-trf (ours) | 38.2 | 47.8 | 51.2 | 44.7 | **55.8** | 47.5 |
| ReMasker | 26.7 | 36.2 | 37.9 | 30.5 | 40.7 | 34.4 |
| HyperImpute | 22.5 | 35.2 | 29.7 | 31.1 | 45.9 | 32.9 |
| MiceForest | 30.7 | 40.4 | 39.8 | 35.2 | 49.0 | 39.0 |
| GAIN | 14.0 | 20.8 | 20.6 | 12.5 | 13.3 | 16.2 |
| EM | 17.4 | 26.8 | 28.2 | 16.7 | 27.9 | 23.4 |
| KNN | 28.5 | 24.9 | 25.4 | 26.3 | 37.2 | 28.5 |
| Naive | 19.6 | 30.9 | 31.0 | 19.6 | 30.1 | 26.3 |

**Letter**, Imp. Acc ↑

| Method | Monotone | Quasi-Monotone | General | Partial-A | Partial-B | Mean |
|---|---:|---:|---:|---:|---:|---:|
| PMAE-mix (ours) | **48.6** | 48.8 | **43.5** | **51.9** | 56.5 | **49.9** |
| PMAE-trf (ours) | 44.8 | **51.0** | 43.4 | 46.3 | **58.2** | 48.8 |
| ReMasker | 31.0 | 29.8 | 28.9 | 32.0 | 30.0 | 30.3 |
| HyperImpute | 33.0 | 35.1 | 29.2 | 37.4 | 43.4 | 35.6 |
| MiceForest | 26.9 | 29.8 | 25.5 | 31.0 | 37.3 | 30.1 |
| GAIN | 17.1 | 22.6 | 20.6 | 25.2 | 20.9 | 21.3 |
| EM | 27.5 | 27.7 | 27.9 | 29.4 | 29.6 | 28.4 |
| KNN | 45.1 | 33.3 | 21.6 | 45.1 | 54.5 | 39.9 |
| Naive | 21.9 | 23.6 | 23.6 | 22.3 | 25.2 | 23.3 |

</details>

## 🔁 Pipeline

1. **Collect** the tables from UCI (Diabetes from scikit-learn).
2. **Split** each table 60/20/20 into train, validation and test.
3. **Mask** train and validation entries with a missing pattern; the test split and the target column stay complete.
4. **Encode** with the observed train entries only: numerical columns are quantile-normal transformed; rare categories are merged and categories are integer encoded.
5. **Impute**: the imputer is fit on the train split and fills the train and validation splits.
6. **Map** the imputations back to the original columns and values, and **evaluate**.

<details>
<summary>Repository structure</summary>

```
pmae/
├── masking.py           # proportional masking (Eq. 11)
├── model.py             # the masked autoencoder: tokens, mixer / transformer blocks, heads, loss
├── imputer.py           # PMAE: scaling, training, imputation (fit / transform)
├── run.py               # one method on one scenario: build, impute, evaluate
├── evaluate.py          # Imp. Acc, R², Acc, RMSE, Utility
├── data/
│   ├── datasets.py      # step 1: download and clean the tables
│   ├── split.py         # step 2: fixed stratified 60/20/20 split
│   ├── patterns.py      # step 3: Monotone, Quasi-Monotone, General, Partial-A, Partial-B; MNAR, MAR, MCAR
│   ├── preprocess.py    # step 4: encoders fit on the observed train entries
│   ├── scenario.py      # steps 2-4 for one scenario, and the map back to data space
│   └── schema.py        # column roles
└── baselines/
    ├── remasker.py      # ReMasker, following the official code
    ├── simple.py        # Naive, KNN
    └── external.py      # EM, GAIN, HyperImpute (hyperimpute); MiceForest (miceforest)
scripts/
├── run_all.sh           # all experiments, then the summary tables
├── make_tables.py       # the tables above
├── plot_overview.py     # the overview figure
└── plot_patterns.py     # the patterns figure
tests/                   # masking, model, data pipeline, metrics
```

</details>

## 📝 Notes on this release

The following updates raise the Imp. Acc of PMAE-mix and PMAE-trf by a few points over the previous implementation; masking, loss normalization and architectures are unchanged:

- Numerical columns are standardized and predicted by a linear head (previously min-max scaled, with a bounded head).
- Categorical columns use learned embeddings and a cross-entropy loss scaled by the column's category entropy, as in [CDTD](https://proceedings.iclr.cc/paper_files/paper/2025/hash/61f4e5747b1b753cb35546b15d981f76-Abstract-Conference.html) (ICLR 2025) (previously integer codes with a squared error).

Monotone and Quasi-Monotone are now nested, as in the paper's definitions; the paper's experiments drew each column independently, and those versions are kept as Partial-A and Partial-B.

## 📖 Citation

```bibtex
@inproceedings{kim2025pmae,
  title     = {To Predict or Not to Predict? Proportionally Masked Autoencoders for Tabular Data Imputation},
  author    = {Kim, Jungkyu and Lee, Kibok and Park, Taeyoung},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {17},
  pages     = {17886--17894},
  year      = {2025},
  doi       = {10.1609/aaai.v39i17.33967}
}
```

## 🙏 Acknowledgements

The masked autoencoder and the ReMasker baseline follow the official [ReMasker](https://github.com/tydusky/remasker) code, with transformer and MLP blocks adapted from [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0). The categorical loss calibration follows [CDTD](https://github.com/muellermarkus/cdtd) (MIT). EM, GAIN and HyperImpute use [hyperimpute](https://github.com/vanderschaarlab/hyperimpute) (MIT), and MiceForest uses [miceforest](https://github.com/AnotherSamWilson/miceforest) (MIT). The data loaders and the downstream evaluation follow [AugMask](https://github.com/normal-kim/AugMask).

Released under the MIT License.
