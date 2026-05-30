"""Experiment 2 -- Minimum Overcomplete Basis via output noise (thesis Ch.4.2.1).

The signature result. With 24 outputs but only 16 causes, the plain network
shares partial bars across outputs. Adding uniform zero-mean Gaussian noise
(sigma=0.01) on the outputs after the non-linearity makes the network settle on
exactly 16 active outputs -- each a whole bar -- while the other 8 weight vectors
decay to zero (Fig.27a). This reproduces the MOB claim.

Run:  python -m experiments.exp02_bars_mob
"""

from __future__ import annotations

import os

from nfnet import (
    BarsData,
    NegativeFeedbackNet,
    Periodic,
    ProgressLogger,
    evaluate_bars_recovery,
    linear_anneal,
    nonlinearities as nl,
    noise as ns,
)
from nfnet.viz import HintonSnapshot

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SIZE = 8
N_INPUTS = SIZE * SIZE
N_OUTPUTS = 24
N_STEPS = 100_000
ETA0 = 0.05
SEED = 1


def run(with_noise: bool):
    tag = "noise" if with_noise else "nonoise"
    print(f"[exp02] soft-threshold, {N_OUTPUTS} outputs, {tag}")
    data = BarsData(size=SIZE, prob=1 / 8, rng=SEED)
    net = NegativeFeedbackNet(
        N_INPUTS, n_outputs=N_OUTPUTS,
        nonlinearity=nl.soft_threshold(tau=1.0, lam=4.0),
        noise=ns.uniform_gaussian(sigma=0.1) if with_noise else None,
        rng=SEED,
    )
    callbacks = [
        Periodic(ProgressLogger(), every=20_000),
        Periodic(HintonSnapshot(OUT, f"exp02_{tag}", grid_shape=(SIZE, SIZE)),
                 every=20_000),
    ]
    net.train(data.sampler(), N_STEPS, eta_schedule=linear_anneal(ETA0),
              callbacks=callbacks)
    score = evaluate_bars_recovery(net.W, size=SIZE)
    print(f"  recovered {score['recovered']}/{2*SIZE} bars, "
          f"active_outputs={score['active_outputs']} (expect ~16 with noise), "
          f"mean_best_sim={score['mean_best_similarity']:.3f}")
    return score


if __name__ == "__main__":
    run(with_noise=False)
    run(with_noise=True)
