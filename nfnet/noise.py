"""Additive output-noise injectors (thesis Ch.4).

Noise is added to the outputs *after* the non-linearity and then fed back through
the weights, so it enters both the reconstruction residual and the weight update.
This introduces the weighted-noise penalty  J' = J + sum_i ||w_i||^2 * sigma_i^2
(thesis eq.28), which drives the network to a Minimum Overcomplete Basis: only as
many outputs respond as there are real causes in the data.

An injector is a callable ``noise(step, n_outputs, rng) -> ndarray`` of length
``n_outputs`` (or a scalar broadcastable to it). Return zeros for no noise.
"""

from __future__ import annotations

import numpy as np


class NoiseInjector:
    def __init__(self, fn, name: str):
        self._fn = fn
        self.name = name

    def __call__(self, step: int, n_outputs: int, rng: np.random.Generator):
        return self._fn(step, n_outputs, rng)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"NoiseInjector({self.name})"


def uniform_gaussian(sigma: float = 0.01) -> NoiseInjector:
    """Same zero-mean Gaussian noise on every output (thesis Ch.4.2.1).

    All weight vectors are penalised equally; only features strong enough to beat
    the noise survive. With 24 outputs on the 16-bar data this yields exactly 16
    active outputs (the MOB result, Fig.27).
    """

    def fn(step, n_outputs, rng):
        return rng.normal(0.0, sigma, size=n_outputs)

    return NoiseInjector(fn, f"uniform_gaussian(sigma={sigma})")


def graduated(sigma_min: float = 0.001, sigma_step: float = 0.001) -> NoiseInjector:
    """Noise that grows linearly with output index (thesis Ch.4.4.1).

    sigma_i = sigma_min + i * sigma_step. Forces features to be learned in the
    low-noise end of the output space, so the number/location of active outputs
    can be controlled.
    """

    def fn(step, n_outputs, rng):
        sigmas = sigma_min + np.arange(n_outputs) * sigma_step
        return rng.normal(0.0, 1.0, size=n_outputs) * sigmas

    return NoiseInjector(fn, f"graduated(min={sigma_min},step={sigma_step})")


def localised(low: float = 0.001, high: float = 0.2, well_centres=(), well_width: int = 3
              ) -> NoiseInjector:
    """High noise everywhere except low-noise 'wells' (thesis Ch.4.4.2).

    Creates modules: features can only settle inside the wells, between regions of
    high-amplitude noise. ``well_centres`` are output indices; ``well_width`` is
    the half-width (in outputs) of each low-noise well.
    """

    def fn(step, n_outputs, rng):
        sigmas = np.full(n_outputs, high)
        idx = np.arange(n_outputs)
        for c in well_centres:
            sigmas[np.abs(idx - c) <= well_width] = low
        return rng.normal(0.0, 1.0, size=n_outputs) * sigmas

    return NoiseInjector(fn, f"localised(low={low},high={high})")


class MutableUniformNoise:
    """Uniform Gaussian output noise with a mutable ``sigma`` (for closed-loop control)."""

    name = "mutable_uniform"

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, step: int, n_outputs: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma, size=n_outputs)


class AdaptiveNoise:
    """Activity-normalised output noise (a count-targeting variant).

    Each output's noise variance scales with that output's own recent activity, so
    that rarely-active outputs receive little noise (and survive on their unique
    reconstruction pull) while busy or duplicate outputs receive more. The activity
    estimate is an exponential moving average (EMA) of the per-output mean of y^2,
    updated by :meth:`observe` from the (pre-noise) outputs during training.

    sigma_i = sigma * sqrt( u_i / mean_j(u_j) ), where u is the activity EMA. Before
    any activity is observed, every output uses the base ``sigma`` (uniform).
    """

    name = "adaptive"

    def __init__(self, sigma: float = 0.1, beta: float = 0.99, eps: float = 1e-8):
        self.sigma = sigma
        self.beta = beta            # EMA decay (closer to 1 = slower)
        self.eps = eps
        self.activity: np.ndarray | None = None   # per-output activity EMA

    def observe(self, Y: np.ndarray) -> None:
        """Update the activity EMA from a batch (or single row) of pre-noise outputs."""
        Y = np.atleast_2d(Y)
        batch_activity = np.mean(Y * Y, axis=0)          # per-output mean of y^2
        if self.activity is None:
            self.activity = batch_activity
        else:
            self.activity = self.beta * self.activity + (1.0 - self.beta) * batch_activity

    def _sigmas(self, n_outputs: int) -> np.ndarray:
        if self.activity is None or self.activity.shape[0] != n_outputs:
            return np.full(n_outputs, self.sigma)
        mean_u = float(np.mean(self.activity)) + self.eps
        scale = np.sqrt((self.activity + self.eps) / mean_u)
        return self.sigma * scale

    def __call__(self, step: int, n_outputs: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n_outputs) * self._sigmas(n_outputs)


class WeightRedundancyNoise:
    """Weight-space redundancy-scaled output noise (a count-targeting variant).

    Redundancy of output i is r_i = max_{j != i} |cos(w_i, w_j)| over the output weight
    vectors, so an output whose weight direction duplicates another active output's gets
    strong noise (and is pruned), while a unique direction gets little noise and survives
    regardless of firing rate. Measured directly from the weights (no sampling noise),
    via :meth:`observe_weights` during training. sigma_i = sigma * r_i; uniform base
    sigma before any observation. Inactive outputs (weight norm below ``active_norm``)
    are treated as non-redundant (low noise) so they are free to grow.
    """

    name = "weight_redundancy"

    def __init__(self, sigma: float = 0.1, beta: float = 0.9, eps: float = 1e-8,
                 active_norm: float = 0.1):
        self.sigma = sigma
        self.beta = beta
        self.eps = eps
        self.active_norm = active_norm
        self.redundancy: np.ndarray | None = None

    def observe_weights(self, W: np.ndarray) -> None:
        """Update the redundancy EMA from the output weight matrix W, shape (M, N)."""
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        Wn = W / np.maximum(norms, self.eps)
        C = np.abs(Wn @ Wn.T)                        # (M, M) absolute cosine
        np.fill_diagonal(C, 0.0)
        inactive = norms[:, 0] < self.active_norm    # near-zero outputs are not redundant
        C[inactive, :] = 0.0                         # inactive rows are not redundant, and
        C[:, inactive] = 0.0                         # cannot make active outputs redundant
        r = C.max(axis=1)
        if self.redundancy is None:
            self.redundancy = r
        else:
            self.redundancy = self.beta * self.redundancy + (1.0 - self.beta) * r

    def _sigmas(self, n_outputs: int) -> np.ndarray:
        if self.redundancy is None or self.redundancy.shape[0] != n_outputs:
            return np.full(n_outputs, self.sigma)
        return self.sigma * self.redundancy

    def __call__(self, step: int, n_outputs: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n_outputs) * self._sigmas(n_outputs)


class RedundancyNoise:
    """Redundancy-scaled output noise (a count-targeting variant).

    Each output's noise variance scales with how much its activations correlate with
    the other outputs, so that an output duplicating a direction already covered by
    another active output receives strong noise (and is pruned), while a unique output
    receives little noise and survives regardless of how rarely it fires. The
    redundancy of output i is r_i = max_{j != i} |corr(y_i, y_j)|, estimated as an EMA
    over training batches via :meth:`observe`. sigma_i = sigma * r_i. Before any
    observation every output uses the base ``sigma`` (uniform).

    Note: activation correlation equals representational redundancy only when the true
    causes are independent; with correlated causes it would also flag genuine
    co-activation.
    """

    name = "redundancy"

    def __init__(self, sigma: float = 0.1, beta: float = 0.9, eps: float = 1e-8):
        self.sigma = sigma
        self.beta = beta            # EMA decay for the redundancy vector
        self.eps = eps
        self.redundancy: np.ndarray | None = None   # per-output redundancy EMA in [0,1]

    def observe(self, Y: np.ndarray) -> None:
        """Update the redundancy EMA from a batch of (pre-noise) outputs, shape (B, M)."""
        Y = np.atleast_2d(Y)
        if Y.shape[0] < 2:
            return
        Yc = Y - Y.mean(axis=0, keepdims=True)
        std = np.sqrt(np.sum(Yc * Yc, axis=0))           # per-column L2 of centred data
        denom = np.outer(std, std)
        cov = Yc.T @ Yc                                  # (M, M) unnormalised covariance
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > self.eps, cov / denom, 0.0)
        np.fill_diagonal(corr, 0.0)                      # exclude self-correlation
        r = np.max(np.abs(corr), axis=1)                 # nearest-neighbour redundancy
        if self.redundancy is None:
            self.redundancy = r
        else:
            self.redundancy = self.beta * self.redundancy + (1.0 - self.beta) * r

    def _sigmas(self, n_outputs: int) -> np.ndarray:
        if self.redundancy is None or self.redundancy.shape[0] != n_outputs:
            return np.full(n_outputs, self.sigma)
        return self.sigma * self.redundancy

    def __call__(self, step: int, n_outputs: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n_outputs) * self._sigmas(n_outputs)
