"""A3 runner: train the torch coder on the scaled SynthSAEBench-style data and score MCC.

Sweeps noise modes at scale on the GPU and appends results to a JSONL ledger. Run, e.g.:

  .venv-a3/Scripts/python.exe -m a3.run_synth --noise-mode uniform --sigma 0.1 \
      --d 768 --n-features 16384 --n-outputs 24576 --correlation 0.0 \
      --n-steps 4000 --batch-size 4096 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from a3.synthgen import TorchSyntheticFeatureData
from a3.torchcoder import (
    AdaptiveNoise, RedundancyNoise, TorchNegativeFeedbackCoder, UniformNoise,
    WeightRedundancyNoise, mcc_recovery, soft_threshold,
)

LEDGER = os.path.join(os.path.dirname(__file__), "ledger_a3.jsonl")


def _make_noise(mode: str, sigma: float):
    if sigma <= 0.0:
        return None
    return {
        "uniform": UniformNoise,
        "adaptive": AdaptiveNoise,
        "redundancy": RedundancyNoise,
        "weight_redundancy": WeightRedundancyNoise,
    }[mode](sigma=sigma)


def run(cfg) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_seed = []
    for seed in cfg.seeds:
        data = TorchSyntheticFeatureData(
            d=cfg.d, n_features=cfg.n_features, correlation=cfg.correlation,
            n_groups=cfg.n_groups, hierarchy=cfg.hierarchy, branching=cfg.branching,
            device=device, seed=seed)
        if cfg.width_control == "l2":
            noise = None
            weight_decay = cfg.weight_decay
        else:
            noise = _make_noise(cfg.noise_mode, cfg.sigma)
            weight_decay = 0.0
        net = TorchNegativeFeedbackCoder(
            cfg.d, cfg.n_outputs, nonlinearity=soft_threshold(cfg.tau, cfg.lam),
            noise=noise, weight_decay=weight_decay, weight_init=cfg.weight_init,
            device=device, seed=seed)
        net.train(data.sample_batch, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
                  eta0=cfg.eta0)
        per_seed.append(mcc_recovery(net.W, data.dictionary))
        del data, net
        if device == "cuda":
            torch.cuda.empty_cache()
    mcc = sum(r["mcc"] for r in per_seed) / len(per_seed)
    active = sum(r["active_outputs"] for r in per_seed) / len(per_seed)
    recovered = sum(r["recovered"] for r in per_seed) / len(per_seed)
    return {"mean_mcc": mcc, "mean_active": active, "mean_recovered": recovered,
            "per_seed": per_seed}


def append_ledger(cfg, result, elapsed):
    row = {"config": vars(cfg), "result": result, "elapsed_s": round(elapsed, 1)}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-mode", default="uniform",
                   choices=["uniform", "adaptive", "redundancy", "weight_redundancy"])
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=4.0)
    p.add_argument("--eta0", type=float, default=0.02)
    p.add_argument("--weight-init", type=float, default=1e-2)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--n-features", type=int, default=16384)
    p.add_argument("--n-outputs", type=int, default=24576)
    p.add_argument("--correlation", type=float, default=0.0)
    p.add_argument("--n-groups", type=int, default=256)
    p.add_argument("--hierarchy", action="store_true")
    p.add_argument("--branching", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rationale", default="")
    p.add_argument("--width-control", default="noise", choices=["noise", "l2"],
                   dest="width_control",
                   help="Width-control mechanism: noise (default) or l2 weight decay")
    p.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay",
                   help="L2 weight-decay coefficient (used when --width-control=l2)")
    cfg = p.parse_args()

    t = time.time()
    result = run(cfg)
    elapsed = time.time() - t
    append_ledger(cfg, result, elapsed)
    mode_tag = (f"l2 wd={cfg.weight_decay}" if cfg.width_control == "l2"
                else f"{cfg.noise_mode} sigma={cfg.sigma}")
    print(f"[{mode_tag} corr={cfg.correlation}] "
          f"mcc={result['mean_mcc']:.3f} active={result['mean_active']:.0f} "
          f"recovered={result['mean_recovered']:.0f}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
