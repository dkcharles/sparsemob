#!/usr/bin/env bash
# Scale confirmation of the noise-vs-L2 identity (d=768, 16384 features, 24576 outputs).
# noise at strength s vs deterministic L2 at weight_decay=s^2 (the Eq.5 mapping).
# Tests whether the bars finding (noise holds the plateau where L2 over-prunes) recurs
# at benchmark scale. Appends to a3/ledger_a3.jsonl.
set -e
PY=.venv-a3/Scripts/python.exe
COMMON="--d 768 --n-features 16384 --n-outputs 24576 --n-steps 2000 --batch-size 4096 --seeds 0 1 2"
for s in 0.15 0.20 0.25; do
  wd=$(python -c "print($s**2)")
  echo "=== noise sigma=$s ==="
  $PY -m a3.run_synth --noise-mode uniform --sigma "$s" $COMMON \
     --rationale "scale identity: noise s=$s" 2>&1 | grep -vE "Warning|warn"
  echo "=== L2 weight_decay=$wd (=$s^2) ==="
  $PY -m a3.run_synth --width-control l2 --weight-decay "$wd" $COMMON \
     --rationale "scale identity: L2 wd=$wd" 2>&1 | grep -vE "Warning|warn"
done
echo "=== SCALE_IDENTITY_DONE ==="
