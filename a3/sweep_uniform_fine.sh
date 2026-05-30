#!/usr/bin/env bash
# Finer uniform-noise sweep over the transition region (0 < sigma < 0.4), to map the
# slide of the active count through the true count (16384) with no plateau, and the
# subsequent collapse to zero. Appends to a3/ledger_a3.jsonl.
set -e
PY=.venv-a3/Scripts/python.exe
for s in 0.05 0.10 0.15 0.30; do
  "$PY" -m a3.run_synth --noise-mode uniform --sigma "$s" \
    --d 768 --n-features 16384 --n-outputs 24576 \
    --n-steps 2000 --batch-size 4096 --seeds 0 1 \
    --rationale "A3 fine transition map" 2>&1 | grep -vE "UserWarning|functional_tensor|cpu = "
done
echo "FINE_SWEEP_DONE"
