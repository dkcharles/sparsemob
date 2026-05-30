"""Experiment 1 -- the bars data headline result (thesis Ch.3.3.1).

Trains the soft-threshold negative feedback network on mixed horizontal/vertical
bars. With 16 outputs (= number of causes) each output should converge to a
single individual bar. For contrast, also trains the plain linear network, which
finds the principal *subspace* (mixtures), not the individual bars.

Run:  python -m experiments.exp01_bars
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
)
from nfnet.viz import HintonSnapshot, save_hinton

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SIZE = 8
N_INPUTS = SIZE * SIZE
N_STEPS = 50_000
ETA0 = 0.05
SEED = 0


def run_softthreshold():
    print("[exp01] soft-threshold network, 16 outputs")
    data = BarsData(size=SIZE, prob=1 / 8, rng=SEED)
    net = NegativeFeedbackNet(
        N_INPUTS, n_outputs=16,
        nonlinearity=nl.soft_threshold(tau=1.0, lam=4.0),
        rng=SEED,
    )
    callbacks = [
        Periodic(ProgressLogger(), every=10_000),
        Periodic(HintonSnapshot(OUT, "exp01_soft", grid_shape=(SIZE, SIZE)),
                 every=10_000),
    ]
    net.train(data.sampler(), N_STEPS, eta_schedule=linear_anneal(ETA0),
              callbacks=callbacks)
    score = evaluate_bars_recovery(net.W, size=SIZE)
    print(f"  recovered {score['recovered']}/{2*SIZE} bars, "
          f"active_outputs={score['active_outputs']}, "
          f"mean_best_sim={score['mean_best_similarity']:.3f}")
    return score


def run_linear():
    print("[exp01] plain linear (PCA subspace) network, 16 outputs")
    data = BarsData(size=SIZE, prob=1 / 8, rng=SEED)
    net = NegativeFeedbackNet(N_INPUTS, n_outputs=16, nonlinearity=nl.identity,
                              rng=SEED)
    net.train(data.sampler(), N_STEPS, eta_schedule=linear_anneal(ETA0))
    save_hinton(net.W, os.path.join(OUT, "exp01_linear_final.png"),
                grid_shape=(SIZE, SIZE), title="exp01 linear subspace (final)")
    print("  saved outputs/exp01_linear_final.png (mixtures, not individual bars)")


if __name__ == "__main__":
    run_linear()
    run_softthreshold()
