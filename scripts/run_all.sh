#!/usr/bin/env bash
# Full ladder on a laptop. Times measured on an Apple M3 / 16 GB with the default config.
#
#   scripts/run_all.sh quick   # 1 seed, no learned tuning          ~25 min
#   scripts/run_all.sh full    # 3 seeds + validation-set tuning    ~1.5 h
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"
case "$MODE" in
  quick) SEEDS=(0) ;;
  full)  SEEDS=(0 1 2) ;;
  *) echo "usage: $0 [quick|full]" >&2; exit 1 ;;
esac
PY="uv run python -m"

# 0. Data (skipped if present): test split + 64 train shards (~1.2k videos, ~8 GB download, ~3 min)
if [[ ! -f data/validation_video.npy || ! -f data/train_video.npy ]]; then
  $PY slotcourse.data.prepare download --split validation
  $PY slotcourse.data.prepare download --split train --max-shards 64
  $PY slotcourse.data.prepare consolidate
fi

# 1. Validation-set tuning of clustering features (~5 min). Copy the winner into configs/default.yaml.
if [[ "$MODE" == "full" ]]; then
  $PY slotcourse.tune clustering --videos 100
fi

# 2. Clustering baselines on test (C1, C3: seconds; C2: ~4 min)
for m in C1 C2 C3; do $PY slotcourse.evaluate --method "$m"; done

# 3. Learned-rung tuning (~16 min): masking fraction for L3, penalty weight for L4
if [[ "$MODE" == "full" ]]; then
  $PY slotcourse.tune learned --variant L3 --param model.mask_ratio --values 0.15 0.3 0.5 --steps 1000
  $PY slotcourse.tune learned --variant L4 --param model.slot_penalty --values 0.01 0.1 1.0 --steps 1000
fi

# 4. Learned rungs: ~5 min train + <1 min eval each
for seed in "${SEEDS[@]}"; do
  for v in L1 L2 L3 L4; do
    $PY slotcourse.train --variant "$v" --seed "$seed"
    $PY slotcourse.evaluate --method "$v" --seed "$seed"
  done
done

# 5. Table, rung differences, length plot, overlays -> results/report/test/
$PY slotcourse.report
