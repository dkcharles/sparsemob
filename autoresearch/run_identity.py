"""Five-axis noise-vs-L2 identity experiment harness.

Tests the Eq.5 prediction: additive output noise with strength s is equivalent
to a deterministic L2 weight penalty with weight_decay = s**2.

Five axes measured:
  1. active_mean / active_std   -- active output count across seeds (noise and l2)
  2. recovery_mean / recovery_std -- cause-recovery fraction across seeds
  3. matched_atom_cos           -- mean |cos| between noise and L2 weight matrices
                                   (same seed/data per pair, Hungarian matched)
  4. trajectory                 -- active count vs step for one seed, mid strength,
                                   both mechanisms (~20 snapshot points)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from nfnet.data import BarsData
from nfnet.synth import SyntheticFeatureData
from nfnet.network import NegativeFeedbackNet, linear_anneal
from nfnet import nonlinearities as nl
from nfnet.noise import uniform_gaussian
from nfnet.compare import active_count, matched_atom_similarity
from nfnet.callbacks import Callback, TrainingState
from autoresearch.objectives import MOBObjective, SynthObjective


class _ActiveTracker(Callback):
    """Record (step, active_count) every ``interval`` steps."""

    def __init__(self, interval: int):
        self.interval = max(1, interval)
        self.records: list[tuple[int, int]] = []

    def on_step(self, state: TrainingState) -> None:
        if state.step % self.interval == 0:
            self.records.append((state.step, active_count(state.weights)))

    def on_end(self, state: TrainingState) -> None:
        # Always record the final state if not already recorded.
        if not self.records or self.records[-1][0] != state.step:
            self.records.append((state.step, active_count(state.weights)))


def identity_trial(
    experiment: str,
    strengths,
    seeds,
    n_steps: int,
    n_outputs: int,
    d: int = 64,
    n_features: int = 128,
    batch_size: int = 256,
    eta0: float = 0.05,
    traj_interval: int | None = None,
) -> dict:
    """Run the five-axis noise-vs-L2 identity experiment.

    Parameters
    ----------
    experiment : "mob" or "synth"
    strengths : sequence of floats
        Noise sigma (and sqrt of weight_decay) values to sweep.
    seeds : sequence of ints
    n_steps : int
    n_outputs : int
    d : int
        Input dimensionality for synth experiment.
    n_features : int
        Number of ground-truth features for synth experiment.
    batch_size : int
        Mini-batch size for synth experiment.
    eta0 : float
        Initial learning rate (linearly annealed to zero).
    traj_interval : int or None
        Override the active-count recording interval for the trajectory.

    Returns
    -------
    dict with keys:
        "strengths"       : list[float]
        "noise"           : {"active_mean", "active_std", "recovery_mean", "recovery_std"}
        "l2"              : same shape as "noise"
        "matched_atom_cos": list[float], one value per strength
        "trajectory"      : {"steps", "noise", "l2"} -- active-count vs step,
                            first seed, mid strength, both mechanisms
    """
    strengths = list(strengths)
    seeds = list(seeds)

    # Non-linearity: soft threshold as used throughout the thesis results.
    soft = nl.soft_threshold(tau=1.0, lam=4.0)

    # Determine trajectory settings: first seed, middle strength index.
    mid_idx = len(strengths) // 2
    interval = traj_interval if traj_interval is not None else max(1, n_steps // 20)

    # --- Build the Eq.5 mapping helper -----------------------------------------
    # For strength s: noise=uniform_gaussian(s), weight_decay=0 (noise net)
    #                 noise=None,               weight_decay=s**2 (L2 net)
    # ---------------------------------------------------------------------------

    def _train_mob(seed: int, strength: float, use_noise: bool, callbacks=()):
        """Train one MOB network; return trained NegativeFeedbackNet."""
        data = BarsData(size=8, prob=1 / 8, rng=seed)
        noise = uniform_gaussian(sigma=strength) if use_noise else None
        wd = 0.0 if use_noise else strength ** 2
        net = NegativeFeedbackNet(
            64, n_outputs=n_outputs, nonlinearity=soft,
            noise=noise, weight_decay=wd, rng=seed,
        )
        net.train(data.sampler(), n_steps,
                  eta_schedule=linear_anneal(eta0), callbacks=callbacks)
        return net

    def _score_mob(net: NegativeFeedbackNet) -> dict:
        mob_result = MOBObjective(size=8).per_seed(net.W)
        return {
            "active": active_count(net.W),
            "recovered": mob_result["recovered"],
        }

    def _train_synth(seed: int, strength: float, use_noise: bool, callbacks=()):
        """Train one synth network; return (net, data) so the dictionary is accessible."""
        data = SyntheticFeatureData(d=d, n_features=n_features, signed=False, rng=seed)
        noise = uniform_gaussian(sigma=strength) if use_noise else None
        wd = 0.0 if use_noise else strength ** 2
        net = NegativeFeedbackNet(
            d, n_outputs=n_outputs, nonlinearity=soft,
            noise=noise, weight_decay=wd, rng=seed,
        )
        net.train_batched(data.sample_batch, n_steps, batch_size=batch_size,
                          eta_schedule=linear_anneal(eta0), callbacks=callbacks)
        return net, data

    def _score_synth(net: NegativeFeedbackNet, data: SyntheticFeatureData) -> dict:
        synth_result = SynthObjective(n_features).per_seed(net.W, data.dictionary)
        return {
            "active": active_count(net.W),
            "recovered": synth_result["recovered"],
        }

    # Accumulate per-mechanism, per-strength lists.
    noise_active: list[list[float]] = [[] for _ in strengths]
    noise_recovery: list[list[float]] = [[] for _ in strengths]
    l2_active: list[list[float]] = [[] for _ in strengths]
    l2_recovery: list[list[float]] = [[] for _ in strengths]
    matched_cos: list[list[float]] = [[] for _ in strengths]

    # Trajectory storage (one seed, mid strength).
    traj_noise_tracker: _ActiveTracker | None = None
    traj_l2_tracker: _ActiveTracker | None = None

    for si, strength in enumerate(strengths):
        is_mid = (si == mid_idx)

        for seed in seeds:
            is_first_seed = (seed == seeds[0])
            want_traj = is_mid and is_first_seed

            if experiment == "mob":
                noise_cbs = (_ActiveTracker(interval),) if want_traj else ()
                l2_cbs = (_ActiveTracker(interval),) if want_traj else ()

                net_noise = _train_mob(seed, strength, use_noise=True, callbacks=noise_cbs)
                net_l2 = _train_mob(seed, strength, use_noise=False, callbacks=l2_cbs)

                ns_score = _score_mob(net_noise)
                l2_score = _score_mob(net_l2)

            elif experiment == "synth":
                noise_cbs = (_ActiveTracker(interval),) if want_traj else ()
                l2_cbs = (_ActiveTracker(interval),) if want_traj else ()

                net_noise, data_noise = _train_synth(seed, strength, use_noise=True,
                                                     callbacks=noise_cbs)
                net_l2, data_l2 = _train_synth(seed, strength, use_noise=False,
                                               callbacks=l2_cbs)
                ns_score = _score_synth(net_noise, data_noise)
                l2_score = _score_synth(net_l2, data_l2)

            else:
                raise ValueError(f"unknown experiment: {experiment!r}")

            if want_traj:
                traj_noise_tracker = noise_cbs[0]
                traj_l2_tracker = l2_cbs[0]

            noise_active[si].append(float(ns_score["active"]))
            noise_recovery[si].append(float(ns_score["recovered"]))
            l2_active[si].append(float(l2_score["active"]))
            l2_recovery[si].append(float(l2_score["recovered"]))

            cos_val = float(matched_atom_similarity(
                net_noise.W if experiment == "mob" else net_noise.W,
                net_l2.W if experiment == "mob" else net_l2.W,
            ))
            matched_cos[si].append(cos_val)

    # Aggregate across seeds for each strength.
    def _agg(values_per_strength):
        means = [float(np.mean(v)) for v in values_per_strength]
        stds = [float(np.std(v)) for v in values_per_strength]
        return means, stds

    na_mean, na_std = _agg(noise_active)
    nr_mean, nr_std = _agg(noise_recovery)
    la_mean, la_std = _agg(l2_active)
    lr_mean, lr_std = _agg(l2_recovery)
    mc_mean = [float(np.mean(v)) for v in matched_cos]

    # Build trajectory output.
    if traj_noise_tracker is not None and traj_noise_tracker.records:
        traj_steps = [r[0] for r in traj_noise_tracker.records]
        traj_noise_vals = [r[1] for r in traj_noise_tracker.records]
    else:
        traj_steps = []
        traj_noise_vals = []

    if traj_l2_tracker is not None and traj_l2_tracker.records:
        traj_l2_vals = [r[1] for r in traj_l2_tracker.records]
    else:
        traj_l2_vals = []

    return {
        "strengths": strengths,
        "noise": {
            "active_mean": na_mean,
            "active_std": na_std,
            "recovery_mean": nr_mean,
            "recovery_std": nr_std,
        },
        "l2": {
            "active_mean": la_mean,
            "active_std": la_std,
            "recovery_mean": lr_mean,
            "recovery_std": lr_std,
        },
        "matched_atom_cos": mc_mean,
        "trajectory": {
            "steps": traj_steps,
            "noise": traj_noise_vals,
            "l2": traj_l2_vals,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Five-axis noise-vs-L2 identity experiment (Eq.5 validation)."
    )
    p.add_argument("--experiment", required=True, choices=["mob", "synth"])
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.0, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-steps", type=int, default=100_000)
    p.add_argument("--n-outputs", type=int, default=24)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eta0", type=float, default=0.05)
    a = p.parse_args()

    rec = identity_trial(
        experiment=a.experiment,
        strengths=a.strengths,
        seeds=a.seeds,
        n_steps=a.n_steps,
        n_outputs=a.n_outputs,
        d=a.d,
        n_features=a.n_features,
        batch_size=a.batch_size,
        eta0=a.eta0,
    )

    # Print per-strength summary.
    print(f"\n{'strength':>10}  {'noise_active':>12}  {'l2_active':>10}")
    for i, s in enumerate(rec["strengths"]):
        print(f"  {s:8.4f}  {rec['noise']['active_mean'][i]:12.2f}  "
              f"{rec['l2']['active_mean'][i]:10.2f}")

    # Write JSON output.
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "results", "identity")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{a.experiment}.json")
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
