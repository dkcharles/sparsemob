"""The negative feedback network (Charles 1999, thesis Ch.2-4).

One small update rule underlies every result in the thesis. With f = identity it
is equivalent to Oja's Subspace algorithm and spans the principal subspace; swap
in a positive non-linearity and/or add output noise and it becomes a sparse,
multiple-cause coder.

Per presentation of input x (eq.13-15, 24-27):
    a = W x                      feedforward
    y = f(a) (+ noise)           non-linearity, optional additive output noise
    e = x - W^T y                feedback residual at the inputs
    W += eta * outer(y, e)       Hebbian learning on the residual

The feedback term self-stabilises learning, so weights need no explicit
normalisation. Optional half-wave rectification of the *weights* gives the
non-negative-weight networks of Ch.2-3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .callbacks import Callback, TrainingState
from .nonlinearities import NonLinearity, identity
from .noise import NoiseInjector


def linear_anneal(eta0: float) -> Callable[[int, int], float]:
    """Learning rate annealed linearly to zero over training (thesis Ch.3-4)."""
    return lambda step, total: eta0 * (1.0 - step / max(total, 1))


def constant_eta(eta0: float) -> Callable[[int, int], float]:
    return lambda step, total: eta0


class NegativeFeedbackNet:
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        nonlinearity: NonLinearity = identity,
        noise: NoiseInjector | None = None,
        nonneg_weights: bool = False,
        weight_init: float = 1e-3,
        weight_decay: float = 0.0,
        act_l1: float = 0.0,
        topk: int | None = None,
        feedback: bool = True,
        rng: np.random.Generator | int | None = None,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.f = nonlinearity
        self.noise = noise
        self.nonneg_weights = nonneg_weights
        self.weight_decay = weight_decay
        self.act_l1 = act_l1
        self.topk = topk
        self.feedback = feedback
        self.rng = np.random.default_rng(rng)

        # Small random initial weights, ~1e-3 (thesis). Non-negative networks
        # start in the positive quadrant.
        if nonneg_weights:
            self.W = self.rng.uniform(0.0, weight_init, size=(n_outputs, n_inputs))
        else:
            self.W = self.rng.uniform(-weight_init, weight_init, size=(n_outputs, n_inputs))

    def _apply_selection(self, Y: np.ndarray) -> np.ndarray:
        """Apply activation-L1 prox and/or top-k selection to a (B, M) output block."""
        if self.act_l1:
            Y = np.sign(Y) * np.maximum(np.abs(Y) - self.act_l1, 0.0)
        if self.topk is not None and self.topk < self.n_outputs:
            keep = np.argsort(np.abs(Y), axis=1)[:, -self.topk:]
            mask = np.zeros_like(Y, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            Y = np.where(mask, Y, 0.0)
        return Y

    def forward(self, x: np.ndarray, add_noise: bool = False, step: int = 0):
        """Return (a, y). Set ``add_noise`` to include output noise."""
        a = self.W @ x
        y = self.f(a)
        if add_noise and self.noise is not None:
            y = y + self.noise(step, self.n_outputs, self.rng)
        return a, y

    def train_step(self, x: np.ndarray, eta: float, step: int = 0) -> np.ndarray:
        a = self.W @ x
        y = self.f(a)
        if self.noise is not None:
            y = y + self.noise(step, self.n_outputs, self.rng)
        y = self._apply_selection(y[None, :])[0]
        e = x - self.W.T @ y if self.feedback else x  # feedback residual
        self.W += eta * np.outer(y, e)       # Hebbian on residual
        if self.weight_decay:
            self.W -= eta * self.weight_decay * self.W
        if self.nonneg_weights:
            np.maximum(self.W, 0.0, out=self.W)
        return y

    def train_step_batch(self, X: np.ndarray, eta: float, step: int = 0) -> np.ndarray:
        """One minibatched update. X is (B, n_inputs); returns outputs Y (B, n_outputs).

        Vectorised form of train_step: A=XWᵀ, Y=f(A)(+noise), E=X−YW, W += η·YᵀE/B.
        """
        A = X @ self.W.T                       # (B, M)
        Y = self.f(A)
        if self.noise is not None:
            if hasattr(self.noise, "observe"):
                self.noise.observe(Y)          # update activity from pre-noise outputs
            if hasattr(self.noise, "observe_weights"):
                self.noise.observe_weights(self.W)   # update weight-space redundancy
            # Per-sample noise; injectors return one (n_outputs,) draw each call.
            Y = Y + np.array([self.noise(step, self.n_outputs, self.rng)
                              for _ in range(X.shape[0])])
        Y = self._apply_selection(Y)
        E = X - Y @ self.W if self.feedback else X  # (B, N)
        self.W += eta * (Y.T @ E) / X.shape[0]  # (M, N)
        if self.weight_decay:
            self.W -= eta * self.weight_decay * self.W
        if self.nonneg_weights:
            np.maximum(self.W, 0.0, out=self.W)
        return Y

    def train_batched(self, sampler_batch, n_steps: int, batch_size: int = 256,
                      eta_schedule=0.05, callbacks=()):
        """Train for n_steps minibatches drawn from ``sampler_batch(batch_size) -> (B, n_inputs)``.

        ``eta_schedule`` is a constant float or a callable ``(step, total) -> eta``.
        """
        eta_fn = constant_eta(float(eta_schedule)) if not callable(eta_schedule) else eta_schedule
        last_y = np.zeros(self.n_outputs)
        for step in range(n_steps):
            eta = eta_fn(step, n_steps)
            X = sampler_batch(batch_size)
            Y = self.train_step_batch(X, eta, step)
            last_y = Y.mean(axis=0)
            if callbacks:
                state = TrainingState(step=step, total_steps=n_steps, eta=eta,
                                      weights=self.W.copy(), outputs=last_y)
                for cb in callbacks:
                    cb.on_step(state)
        end_state = TrainingState(step=n_steps, total_steps=n_steps,
                                  eta=eta_fn(n_steps, n_steps),
                                  weights=self.W.copy(), outputs=last_y)
        for cb in callbacks:
            cb.on_end(end_state)
        return self

    def train(
        self,
        sampler: Callable[[], np.ndarray],
        n_steps: int,
        eta_schedule: Callable[[int, int], float] | float = 0.05,
        callbacks: Sequence[Callback] = (),
    ) -> "NegativeFeedbackNet":
        """Train for ``n_steps`` presentations drawn from ``sampler()``.

        ``eta_schedule`` is either a constant float or a function
        ``(step, total) -> eta`` (see :func:`linear_anneal`).
        """
        if not callable(eta_schedule):
            eta_fn = constant_eta(float(eta_schedule))
        else:
            eta_fn = eta_schedule

        last_y = np.zeros(self.n_outputs)
        for step in range(n_steps):
            eta = eta_fn(step, n_steps)
            x = sampler()
            last_y = self.train_step(x, eta, step)
            if callbacks:
                state = TrainingState(
                    step=step, total_steps=n_steps, eta=eta,
                    weights=self.W.copy(), outputs=last_y,
                )
                for cb in callbacks:
                    cb.on_step(state)

        end_state = TrainingState(
            step=n_steps, total_steps=n_steps, eta=eta_fn(n_steps, n_steps),
            weights=self.W.copy(), outputs=last_y,
        )
        for cb in callbacks:
            cb.on_end(end_state)
        return self
