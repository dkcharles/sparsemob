#!/usr/bin/env bash
# Minimal redundancy-misfire confirmation at 16k scale.
#
# A2 (d=64) showed activation-redundancy noise misfires under correlation: it reads
# co-firing of distinct-but-correlated features as redundancy and culls them, whereas
# weight-redundancy (cosine of weight directions) does not. This checks the signature
# at SynthSAEBench scale: for each mode we compare independent (corr=0) vs grouped
# co-firing (corr=0.8) at a base sigma inside the surviving window (uniform sigma=0.2
# left 14908/24576 outputs active, so redundancy-scaled noise stays well clear of the
# collapse cliff at sigma~0.4). The misfire signature is: activation-redundancy drops
# notably more active outputs / recovers fewer features under correlation than
# independent, while weight-redundancy is roughly unchanged. Appends to ledger_a3.jsonl.
set -e
PY=.venv-a3/Scripts/python.exe
for mode in redundancy weight_redundancy; do
  for corr in 0.0 0.8; do
    "$PY" -m a3.run_synth --noise-mode "$mode" --sigma 0.2 \
      --d 768 --n-features 16384 --n-outputs 24576 \
      --correlation "$corr" --n-groups 256 \
      --n-steps 2000 --batch-size 4096 --seeds 0 \
      --rationale "A3 misfire-at-scale: $mode corr=$corr" 2>&1 \
      | grep -vE "UserWarning|functional_tensor|cpu = "
  done
done
echo "MISFIRE_SWEEP_DONE"
