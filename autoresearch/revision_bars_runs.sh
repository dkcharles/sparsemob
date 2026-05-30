#!/usr/bin/env bash
# Bars-based revision experiments (cheap; run alongside the synth identity sweep).
#   Task 10 (A1b): output noise with feedback OFF -> predicted inert (active flat in sigma)
#   Task 12 (A2):  noise-floor sweep at 20 seeds for a usable CI on the sharp floor
#   Task 11 (A1c): L1 and top-k width-control baselines on bars
# All append to autoresearch/ledger/experiments.jsonl.
set -e
PY=python

echo "=== Task 10: A1b no-feedback noise sweep (bars) ==="
for s in 0.0 0.1 0.2 0.3 0.4; do
  $PY -m autoresearch.run_trial --experiment mob --sigma "$s" --no-feedback \
    --n-outputs 24 --n-steps 100000 --seeds 0 1 2 3 4 \
    --rationale "A1b: output noise inert without feedback"
done

echo "=== Task 12: noise-floor sweep, 20 seeds (bars) ==="
for s in 0.0 0.05 0.075 0.1 0.125 0.15 0.2 0.3 0.4 0.5; do
  $PY -m autoresearch.run_trial --experiment mob --sigma "$s" \
    --n-outputs 24 --n-steps 100000 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --rationale "A2 floor 20 seeds"
done

echo "=== Task 11: L1 width-control sweep (bars) ==="
for l in 0.05 0.1 0.2 0.3; do
  $PY -m autoresearch.run_trial --experiment mob --sigma 0.0 --width-control l1 --act-l1 "$l" \
    --n-outputs 24 --n-steps 100000 --seeds 0 1 2 3 4 --rationale "A1c L1 bars"
done

echo "=== Task 11: top-k width-control sweep (bars) ==="
for k in 12 16 20 24; do
  $PY -m autoresearch.run_trial --experiment mob --sigma 0.0 --width-control topk --topk "$k" \
    --n-outputs 24 --n-steps 100000 --seeds 0 1 2 3 4 --rationale "A1c topk bars"
done

echo "=== BARS_RUNS_DONE ==="
