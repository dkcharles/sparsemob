"""Run one knob-config trial: train nfnet across seeds, score the objective,
and record the result in the append-only ledger (with a keep/discard decision).

The mechanism per experiment is fixed here; only TrialConfig knobs vary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from nfnet.data import BarsData, SignedBarsData
from nfnet.synth import SyntheticFeatureData
from nfnet.network import NegativeFeedbackNet, linear_anneal
from nfnet import nonlinearities as nl
from nfnet import noise as ns
from nfnet.noise import AdaptiveNoise, MutableUniformNoise, RedundancyNoise, WeightRedundancyNoise
from nfnet.callbacks import CountController
from .objectives import MOBObjective, ObjectiveResult, SignedAbsObjective, SignedBarsObjective, SynthObjective


@dataclass
class TrialConfig:
    experiment: str
    tau: float = 1.0
    lam: float = 4.0
    eta0: float = 0.05
    sigma: float = 0.1
    n_outputs: int = 24
    weight_init: float = 1e-3
    n_steps: int = 100_000
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    d: int = 128
    n_features: int = 512
    batch_size: int = 256
    noise_mode: str = "uniform"   # synth experiment: uniform | adaptive | controller
    correlation: float = 0.0      # synth firing structure: grouped co-firing strength
    n_groups: int = 16            # synth: number of correlated feature groups
    hierarchy: bool = False       # synth: parent-gated child firing
    branching: int = 4            # synth: hierarchy tree branching factor
    width_control: str = "noise"  # noise | l2 | l1 | topk
    weight_decay: float = 0.0     # L2 strength when width_control == "l2"
    act_l1: float = 0.0           # L1 strength when width_control == "l1"
    topk: int | None = None       # k when width_control == "topk"
    feedback: bool = True         # A1b control


SIZE = 8
N_INPUTS = SIZE * SIZE


def _net_kwargs(cfg: TrialConfig) -> dict:
    return dict(
        weight_decay=cfg.weight_decay if cfg.width_control == "l2" else 0.0,
        act_l1=cfg.act_l1 if cfg.width_control == "l1" else 0.0,
        topk=cfg.topk if cfg.width_control == "topk" else None,
        feedback=cfg.feedback,
    )


def _make_net(cfg: TrialConfig, seed: int, use_noise: bool, nonlinearity) -> NegativeFeedbackNet:
    return NegativeFeedbackNet(
        N_INPUTS, n_outputs=cfg.n_outputs, nonlinearity=nonlinearity,
        noise=ns.uniform_gaussian(sigma=cfg.sigma) if use_noise else None,
        weight_init=cfg.weight_init, rng=seed,
        **_net_kwargs(cfg),
    )


def _train(cfg: TrialConfig, data, seed: int, use_noise: bool, nonlinearity) -> np.ndarray:
    # `data` is duck-typed: any object exposing .sampler() (BarsData / SignedBarsData).
    net = _make_net(cfg, seed, use_noise, nonlinearity)
    net.train(data.sampler(), cfg.n_steps, eta_schedule=linear_anneal(cfg.eta0))
    return net.W


def run_trial(cfg: TrialConfig) -> ObjectiveResult:
    # Explicit branches by design (YAGNI at three experiments); refactor to a
    # registry only if the count keeps growing.
    use_noise = cfg.width_control == "noise" and cfg.sigma > 0.0
    soft = nl.soft_threshold(tau=cfg.tau, lam=cfg.lam)

    if cfg.experiment == "mob":
        objective = MOBObjective(size=SIZE)
        per_seed = []
        for seed in cfg.seeds:
            data = BarsData(size=SIZE, prob=1 / 8, rng=seed)
            per_seed.append(objective.per_seed(_train(cfg, data, seed, use_noise, soft)))
        return objective.aggregate(per_seed)

    if cfg.experiment == "signed_bars":
        objective = SignedBarsObjective(size=SIZE)
        per_seed = []
        for seed in cfg.seeds:
            data = SignedBarsData(size=SIZE, prob=1 / 8, rng=seed)
            per_seed.append(objective.per_seed(_train(cfg, data, seed, use_noise, soft)))
        return objective.aggregate(per_seed)

    if cfg.experiment == "signed_bars_signed":
        objective = SignedAbsObjective(size=SIZE)
        shrink = nl.soft_shrink(tau=cfg.tau, lam=cfg.lam)
        per_seed = []
        for seed in cfg.seeds:
            data = SignedBarsData(size=SIZE, prob=1 / 8, rng=seed)
            per_seed.append(objective.per_seed(_train(cfg, data, seed, use_noise, shrink)))
        return objective.aggregate(per_seed)

    if cfg.experiment == "synth":
        objective = SynthObjective(n_features=cfg.n_features)
        per_seed = []
        for seed in cfg.seeds:
            data = SyntheticFeatureData(d=cfg.d, n_features=cfg.n_features,
                                        signed=False, correlation=cfg.correlation,
                                        n_groups=cfg.n_groups, hierarchy=cfg.hierarchy,
                                        branching=cfg.branching, rng=seed)
            callbacks = ()
            if not use_noise:
                noise = None
            elif cfg.noise_mode == "adaptive":
                noise = AdaptiveNoise(sigma=cfg.sigma)
            elif cfg.noise_mode == "controller":
                noise = MutableUniformNoise(sigma=cfg.sigma)
                callbacks = (CountController(noise, target=cfg.n_features),)
            elif cfg.noise_mode == "redundancy":
                noise = RedundancyNoise(sigma=cfg.sigma)
            elif cfg.noise_mode == "weight_redundancy":
                noise = WeightRedundancyNoise(sigma=cfg.sigma)
            else:  # uniform
                noise = ns.uniform_gaussian(sigma=cfg.sigma)
            net = NegativeFeedbackNet(
                cfg.d, n_outputs=cfg.n_outputs, nonlinearity=soft, noise=noise,
                weight_init=cfg.weight_init, rng=seed,
                **_net_kwargs(cfg),
            )
            net.train_batched(data.sample_batch, cfg.n_steps, batch_size=cfg.batch_size,
                              eta_schedule=linear_anneal(cfg.eta0), callbacks=callbacks)
            per_seed.append(objective.per_seed(net.W, data.dictionary))
        return objective.aggregate(per_seed)

    raise ValueError(f"unknown experiment: {cfg.experiment!r}")


DEFAULT_LEDGER = os.path.join(os.path.dirname(__file__), "ledger")
DEFAULT_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

MARGIN_TOLERANCE = 1e-9  # worst-case (max per-seed) score must not regress to keep


def default_baseline_path(experiment: str) -> str:
    """Per-experiment baseline file so different experiments never compare scores."""
    return os.path.join(DEFAULT_CONFIGS_DIR, f"{experiment}_baseline.json")


def load_baseline_result(baseline_path: str):
    """Return the kept baseline's result dict, or None if absent/unset."""
    if not os.path.exists(baseline_path):
        return None
    with open(baseline_path) as f:
        data = json.load(f)
    return data.get("result")


def decide_keep(result: ObjectiveResult, cfg: TrialConfig, baseline) -> bool:
    if baseline is None:
        return True
    eps = 1e-9
    # Worst-case guard (spec): never keep a config whose worst seed regresses
    # past a tolerance, even if the mean improves.
    if "margin" in baseline and result.margin > baseline["margin"] + MARGIN_TOLERANCE:
        return False
    if result.seed_pass_fraction > baseline["seed_pass_fraction"] + eps:
        return True
    if abs(result.seed_pass_fraction - baseline["seed_pass_fraction"]) <= eps:
        if result.primary_score < baseline["primary_score"] - eps:
            return True
        if (abs(result.primary_score - baseline["primary_score"]) <= eps
                and cfg.n_steps < baseline["n_steps"]):
            return True
    return False


def append_ledger(cfg: TrialConfig, result: ObjectiveResult, rationale: str,
                  ledger_dir: str = DEFAULT_LEDGER,
                  baseline_path: str | None = None) -> bool:
    os.makedirs(ledger_dir, exist_ok=True)
    if baseline_path is None:
        baseline_path = default_baseline_path(cfg.experiment)
    baseline = load_baseline_result(baseline_path)
    kept = decide_keep(result, cfg, baseline)
    result_summary = {
        "primary_score": result.primary_score,
        "passed": result.passed,
        "margin": result.margin,
        "seed_pass_fraction": result.seed_pass_fraction,
        "diagnostics": result.diagnostics,
    }
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": cfg.experiment,
        "config": asdict(cfg),
        "result": result_summary,
        "rationale": rationale,
        "kept": kept,
    }
    with open(os.path.join(ledger_dir, "experiments.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(os.path.join(ledger_dir, "log.md"), "a") as f:
        f.write(
            f"- [{row['timestamp']}] **{cfg.experiment}** "
            f"score={result.primary_score:.3f} pass={result.seed_pass_fraction:.2f} "
            f"steps={cfg.n_steps} sigma={cfg.sigma} n_out={cfg.n_outputs} "
            f"-> {'KEPT' if kept else 'discard'} | {rationale}\n"
        )
    if kept:
        baseline_dir = os.path.dirname(baseline_path)
        if baseline_dir:
            os.makedirs(baseline_dir, exist_ok=True)
        # Store only the keys decide_keep needs; full result lives in the ledger.
        with open(baseline_path, "w") as f:
            json.dump({"config": asdict(cfg),
                       "result": {"primary_score": result.primary_score,
                                  "seed_pass_fraction": result.seed_pass_fraction,
                                  "n_steps": cfg.n_steps,
                                  "margin": result.margin}}, f, indent=2)
    return kept


def _cli():
    p = argparse.ArgumentParser(description="Run one nfnet autoresearch trial.")
    p.add_argument("--experiment", required=True, choices=["mob", "signed_bars", "signed_bars_signed", "synth"])
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=4.0)
    p.add_argument("--eta0", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-outputs", type=int, default=24)
    p.add_argument("--weight-init", type=float, default=1e-3)
    p.add_argument("--n-steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--n-features", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--noise-mode", default="uniform", choices=["uniform", "adaptive", "controller", "redundancy", "weight_redundancy"])
    p.add_argument("--correlation", type=float, default=0.0)
    p.add_argument("--n-groups", type=int, default=16)
    p.add_argument("--hierarchy", action="store_true")
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--width-control", default="noise", choices=["noise", "l2", "l1", "topk"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--act-l1", type=float, default=0.0)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--no-feedback", dest="feedback", action="store_false", default=True)
    p.add_argument("--rationale", default="(none given)")
    a = p.parse_args()
    cfg = TrialConfig(experiment=a.experiment, tau=a.tau, lam=a.lam, eta0=a.eta0,
                      sigma=a.sigma, n_outputs=a.n_outputs, weight_init=a.weight_init,
                      n_steps=a.n_steps, seeds=a.seeds,
                      d=a.d, n_features=a.n_features, batch_size=a.batch_size,
                      noise_mode=a.noise_mode, correlation=a.correlation,
                      n_groups=a.n_groups, hierarchy=a.hierarchy, branching=a.branching,
                      width_control=a.width_control, weight_decay=a.weight_decay,
                      act_l1=a.act_l1, topk=a.topk, feedback=a.feedback)
    res = run_trial(cfg)
    kept = append_ledger(cfg, res, rationale=a.rationale)
    print(f"score={res.primary_score:.3f} pass_fraction={res.seed_pass_fraction:.2f} "
          f"passed={res.passed} -> {'KEPT' if kept else 'discard'}")
    print("diagnostics:", res.diagnostics)


if __name__ == "__main__":
    _cli()
