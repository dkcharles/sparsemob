#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-.venv-a3/Scripts/python.exe}"
for s in 0.0 0.2 0.4 0.6 0.8 1.2; do
  $PY -m a3.run_synth --noise-mode uniform --sigma $s --d 768 --n-features 16384 \
    --n-outputs 24576 --n-steps 2000 --batch-size 4096 --seeds 0 1 \
    --rationale "A3 count-selection uniform sweep" 2>&1 | grep -vE "UserWarning|functional_tensor|cpu = "
done
echo "SWEEP_DONE"
