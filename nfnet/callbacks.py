"""Training callbacks and the state-snapshot protocol.

This module is intentionally free of matplotlib (or any UI/rendering) so that it
stays portable. The training loop emits a :class:`TrainingState` to each
callback; a future real-time viewer -- in matplotlib, a browser, or a port to
another language -- subscribes to the same interface without the algorithm
needing to change. ``TrainingState.weights`` is a plain ``(n_outputs, n_inputs)``
NumPy array, which is trivial to serialise (e.g. to JSON or a binary frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingState:
    step: int           # current training step (0-based)
    total_steps: int
    eta: float          # learning rate in effect this step
    weights: np.ndarray  # copy of W, shape (n_outputs, n_inputs)
    outputs: np.ndarray  # last output activations y, shape (n_outputs,)


class Callback:
    """Base class. Override either method; defaults are no-ops."""

    def on_step(self, state: TrainingState) -> None:
        ...

    def on_end(self, state: TrainingState) -> None:
        ...


class Periodic(Callback):
    """Fire a wrapped callback every ``every`` steps (plus once at the end)."""

    def __init__(self, inner: Callback, every: int):
        self.inner = inner
        self.every = every

    def on_step(self, state: TrainingState) -> None:
        if self.every > 0 and state.step % self.every == 0:
            self.inner.on_step(state)

    def on_end(self, state: TrainingState) -> None:
        self.inner.on_end(state)


class StateRecorder(Callback):
    """Keep weight snapshots in memory (handy for tests and offline animation)."""

    def __init__(self):
        self.snapshots: list[tuple[int, np.ndarray]] = []

    def on_step(self, state: TrainingState) -> None:
        self.snapshots.append((state.step, state.weights))

    def on_end(self, state: TrainingState) -> None:
        self.snapshots.append((state.step, state.weights))


class ProgressLogger(Callback):
    """Print a one-line progress update."""

    def on_step(self, state: TrainingState) -> None:
        active = int(np.sum(np.linalg.norm(state.weights, axis=1) > 0.1))
        print(
            f"  step {state.step:>7}/{state.total_steps}  "
            f"eta={state.eta:.4f}  active_outputs={active}"
        )

    def on_end(self, state: TrainingState) -> None:
        self.on_step(state)


class CountController(Callback):
    """Closed-loop controller that drives the active-output count toward a target.

    Every ``every`` steps it reads the number of active outputs (weight norm above
    ``active_norm``) and nudges the noise injector's global ``sigma``: raising it when
    too many outputs are active, lowering it when too few. A contrast baseline for the
    activity-normalised AdaptiveNoise: it targets the count directly via a single
    global noise level.
    """

    def __init__(self, injector, target: int, every: int = 200, factor: float = 1.1,
                 active_norm: float = 0.1, sigma_min: float = 1e-3, sigma_max: float = 2.0):
        self.injector = injector
        self.target = target
        self.every = every
        self.factor = factor
        self.active_norm = active_norm
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def on_step(self, state: "TrainingState") -> None:
        if self.every <= 0 or state.step % self.every != 0:
            return
        active = int(np.sum(np.linalg.norm(state.weights, axis=1) > self.active_norm))
        if active > self.target:
            self.injector.sigma *= self.factor
        elif active < self.target:
            self.injector.sigma /= self.factor
        self.injector.sigma = float(min(max(self.injector.sigma, self.sigma_min), self.sigma_max))
