#!/usr/bin/env bash
# Serial GPU re-run of the scale sweeps affected by the code fixes: count-selection
# (uniform; #4 seeding), the fine transition map, the misfire sweep (#2 weight-redundancy
# column mask, #4), and the noise-vs-L2 scale identity (#1). All append to a3/ledger_a3.jsonl;
# figure selectors keep the latest row per key.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY="${PY:-.venv-a3/Scripts/python.exe}"

bash "$HERE/sweep_uniform.sh"
bash "$HERE/sweep_uniform_fine.sh"
bash "$HERE/sweep_misfire.sh"
bash "$HERE/scale_identity.sh"
echo "REFIX_GPU_ALL_DONE"
