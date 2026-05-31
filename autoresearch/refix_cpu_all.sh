#!/usr/bin/env bash
# Serial CPU re-run of everything affected by the code fixes (so the cores are not
# thrashed by parallel jobs): weight-redundancy synth runs (#2), then the noise-vs-L2
# identity sweeps (#1, L2 now a simultaneous update). Overwrites docs/results/identity/*.json.
set -euo pipefail
PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "########## 1/3 weight-redundancy synth re-run ##########"
PY="$PY" bash "$HERE/refix_weight_redundancy.sh"

echo "########## 2/3 identity: bars (mob) ##########"
$PY -m autoresearch.run_identity --experiment mob \
  --strengths 0.0 0.05 0.075 0.1 0.15 0.2 0.3 0.4 \
  --seeds 0 1 2 3 4 --n-steps 100000 --n-outputs 24

echo "########## 3/3 identity: synth ##########"
# n-outputs 192 (over-provisioned, matching the count-selection synth and the original
# synth-identity run); the active count slides through the true feature count of 128.
$PY -m autoresearch.run_identity --experiment synth \
  --strengths 0.0 0.1 0.2 0.3 0.4 0.5 \
  --seeds 0 1 2 3 4 --n-steps 100000 --n-outputs 192

echo "REFIX_CPU_ALL_DONE"
