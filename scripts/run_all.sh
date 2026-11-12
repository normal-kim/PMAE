#!/usr/bin/env bash
# All experiments: 4 datasets x 3 missing patterns (MNAR) x 3 seeds x 9 methods, then the summary tables.
#   bash scripts/run_all.sh [gpu] [dataset ...]    e.g. run_all.sh 1 adult letter, to split the work across GPUs
set -euo pipefail
GPU="${1:-0}"
shift $(( $# > 0 ? 1 : 0 ))
DATASETS="${*:-diabetes shoppers adult letter}"
PATTERNS="monotone quasi general"
SEEDS="0 1 2"
METHODS="pmae-mix pmae-trf remasker hyperimpute miceforest gain em knn naive"

for d in $DATASETS; do for p in $PATTERNS; do for s in $SEEDS; do for method in $METHODS; do
  setting="$p"; [ "$p" = monotone ] && setting="monotone0.6"
  [ -f "runs/$d/$setting-mnar/seed$s/$method.json" ] && continue
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 python -m pmae.run --dataset "$d" --pattern "$p" --seed "$s" --method "$method"
done; done; done; done
python scripts/make_tables.py --runs runs
