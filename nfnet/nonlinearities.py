"""Output non-linearities f(a) for the negative feedback network.

Each non-linearity is a plain callable ``f(a) -> y`` operating element-wise on a
NumPy array. The negative feedback learning rule never needs the derivative of
f (it learns on the input-side residual), so only the forward map is required.

Naming follows the thesis chapters:
  - identity      : linear PCA / subspace network          (Ch.2, eq.13-15)
  - rectify       : half-wave rectification [a]_+          (Ch.3, eq.24-27)
  - exp_shifted   : EXP network                            (Ch.3.2.3)
  - sigmoid_shifted, soft_threshold, logistic_shifted      (Ch.3.3)
  - sinh, square  : comparison baselines (non-linear PCA / EPP)
"""

from __future__ import annotations

import numpy as np


class NonLinearity:
    """Wraps an element-wise function with a readable name (used by plots/logs)."""

    def __init__(self, fn, name: str):
        self._fn = fn
        self.name = name

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return self._fn(a)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"NonLinearity({self.name})"


identity = NonLinearity(lambda a: a, "identity")

rectify = NonLinearity(lambda a: np.maximum(a, 0.0), "rectify")

# EXP network: exp(a-1) - exp(-1). Close to a rectified-linear ramp with a
# slight upturn past 1; the best clean-data performer in the thesis (Ch.3.2.3).
exp_shifted = NonLinearity(
    lambda a: np.exp(a - 1.0) - np.exp(-1.0), "exp_shifted"
)

# Shifted logistic used as an approximate rectifier (thesis shift of 4).
sigmoid_shifted = NonLinearity(
    lambda a: 1.0 / (1.0 + np.exp(4.0 - a)), "sigmoid_shifted"
)

sinh = NonLinearity(np.sinh, "sinh")

square = NonLinearity(lambda a: a ** 2, "square")  # EPP skewness index (y^2)


def soft_threshold(tau: float = 1.0, lam: float = 4.0) -> NonLinearity:
    """Soft-threshold function  y = log(1 + exp(lam*(a - tau))) / lam  (Ch.3.3).

    ``tau`` shifts the threshold along the activation axis; ``lam`` controls the
    gradient. Because the function has no hard cut-off, weights can grow from a
    near-zero start (a hard shifted rectifier cannot). The thesis' most robust
    and easiest-to-tune non-linearity.

    The defaults give a sharp threshold with a negligible output floor
    f(0) = log(1+exp(-lam*tau))/lam ~ 0.004; a larger floor leaves every output
    weakly active and prevents redundant outputs from decaying under the noise
    penalty (see the MOB experiment).
    """

    def fn(a: np.ndarray) -> np.ndarray:
        # log1p(exp(z)) computed stably for large z via np.logaddexp(0, z).
        return np.logaddexp(0.0, lam * (a - tau)) / lam

    return NonLinearity(fn, f"soft_threshold(tau={tau},lam={lam})")


def soft_shrink(tau: float = 1.0, lam: float = 4.0) -> NonLinearity:
    """Sign-preserving soft threshold: y = sign(a) * log(1 + exp(lam*(|a| - tau))) / lam.

    The odd-symmetric counterpart of :func:`soft_threshold`. It squashes small-magnitude
    activations toward zero while preserving sign for large ones, so a single atom can
    represent both the +bar and -bar of a bidirectional cause via positive/negative
    outputs (the sign-preserving alternative to the non-negative code).
    """

    def fn(a: np.ndarray) -> np.ndarray:
        return np.sign(a) * np.logaddexp(0.0, lam * (np.abs(a) - tau)) / lam

    return NonLinearity(fn, f"soft_shrink(tau={tau},lam={lam})")


def logistic_shifted(tau: float = 0.5, lam: float = 4.0) -> NonLinearity:
    """Shifted logistic  y = 1 / (1 + exp(lam*(tau - a)))  (Ch.3.3).

    Saturating sibling of the soft threshold; preferred on large image data sets
    because saturation avoids floating-point overflow.
    """

    def fn(a: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(lam * (tau - a)))

    return NonLinearity(fn, f"logistic_shifted(tau={tau},lam={lam})")
