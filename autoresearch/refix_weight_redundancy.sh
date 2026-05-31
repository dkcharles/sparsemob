#!/usr/bin/env bash
# Re-run the synth weight-redundancy runs that changed after the column-masking fix
# (WeightRedundancyNoise now ignores inactive columns). Feeds count_selection.pdf
# (corr=0 sweep) and a2_structure.png (corr=0 vs corr=0.8 misfire). Appends fresh rows
# to autoresearch/ledger/experiments.jsonl; figure selectors keep the latest row per key.
set -euo pipefail
PY="${PY:-python}"
COMMON="--experiment synth --noise-mode weight_redundancy --n-outputs 192 --d 64 \
  --n-features 128 --n-steps 5000 --batch-size 128 --eta0 0.02 --weight-init 0.01 \
  --seeds 0 1 2"

echo "=== count-selection sweep (independent, corr=0) ==="
for s in 0.1 0.2 0.3 0.4 0.6 0.8 1.2; do
  $PY -m autoresearch.run_trial $COMMON --sigma "$s" --correlation 0.0 \
    --rationale "R3 weight-redundancy sweep sigma=$s (post column-mask fix)"
done

echo "=== misfire: correlated (corr=0.8) ==="
for s in 0.1 0.2 0.3; do
  $PY -m autoresearch.run_trial $COMMON --sigma "$s" --correlation 0.8 --n-groups 16 \
    --rationale "A2 misfire corr wt-red s=$s (post column-mask fix)"
done

echo "WEIGHT_REDUNDANCY_REFIX_DONE"
