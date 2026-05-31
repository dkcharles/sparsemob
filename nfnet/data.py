"""Data generators and evaluation for the thesis experiments.

The bars data (Foldiak / Rumelhart-Zipser) is the headline testbed: an NxN grid
of independently chosen horizontal and vertical bars (thesis Ch.1.2.2). Also
included: signed (bidirectional) bars and stereo-disparity data (Ch.4.6), plus
quantitative bar-recovery metrics so experiment success is verifiable without
eyeballing plots.
"""

from __future__ import annotations

import numpy as np


class BarsData:
    """Random mixtures of horizontal/vertical bars on an ``size x size`` grid.

    Each of the ``2*size`` possible bars is drawn independently with probability
    ``prob`` (thesis default 1/8 on an 8x8 grid -> 16 bars, 64 inputs). Pixels are
    binary; an overlapping H/V crossing is simply 1.
    """

    def __init__(self, size: int = 8, prob: float = 1 / 8, mix: bool = True,
                 rng: np.random.Generator | int | None = None):
        self.size = size
        self.prob = prob
        self.mix = mix  # if False, a sample is all-horizontal or all-vertical
        self.rng = np.random.default_rng(rng)
        self.n_inputs = size * size
        self.n_bars = 2 * size

    def sample(self) -> np.ndarray:
        s = self.size
        grid = np.zeros((s, s))
        if self.mix:
            h = self.rng.random(s) < self.prob
            v = self.rng.random(s) < self.prob
        elif self.rng.random() < 0.5:
            h = self.rng.random(s) < self.prob
            v = np.zeros(s, dtype=bool)
        else:
            h = np.zeros(s, dtype=bool)
            v = self.rng.random(s) < self.prob
        grid[h, :] = 1.0
        grid[:, v] = 1.0
        return grid.ravel()

    def sampler(self):
        return self.sample

    def bar_templates(self) -> np.ndarray:
        """The ``2*size`` ground-truth bar patterns, each as a flat unit vector."""
        s = self.size
        templates = []
        for i in range(s):                 # horizontal bars
            g = np.zeros((s, s)); g[i, :] = 1.0
            templates.append(g.ravel())
        for j in range(s):                 # vertical bars
            g = np.zeros((s, s)); g[:, j] = 1.0
            templates.append(g.ravel())
        T = np.array(templates)
        return T / np.linalg.norm(T, axis=1, keepdims=True)


def evaluate_bars_recovery(W: np.ndarray, size: int = 8, threshold: float = 0.9,
                           active_norm: float = 0.1):
    """Score how well weight vectors recovered the individual bars.

    Active outputs are those whose *raw* weight-vector norm exceeds ``active_norm``
    (the same definition used by :func:`nfnet.compare.active_count`), so the active
    set is consistent across metrics. Recovery is reported two ways:

    Returns a dict with:
      - recovered: bars matched by some active output under the positive-part
        (rectified) cosine >= ``threshold``. This is the primary/historical metric
        for the non-negative bar code.
      - recovered_raw: bars matched under the raw (un-rectified) cosine >=
        ``threshold``; stricter, since it penalises any off-bar (negative) structure.
      - active_outputs: number of active outputs (raw norm > ``active_norm``).
      - mean_best_similarity: average best-match positive-part cosine over all bars.
      - mean_best_similarity_raw: average best-match raw cosine over all bars.
    """
    s = size
    templates = BarsData(size=s).bar_templates()       # (2s, N)
    norms = np.linalg.norm(W, axis=1)                  # raw norm (matches compare.active_count)
    active = norms > active_norm
    Wa = W[active]
    if Wa.shape[0] == 0:
        return {"recovered": 0, "recovered_raw": 0, "active_outputs": 0,
                "mean_best_similarity": 0.0, "mean_best_similarity_raw": 0.0}
    # Positive-part match: the historical protocol for the non-negative bar code.
    Wpos = np.maximum(Wa, 0.0)
    pos = Wpos / np.maximum(np.linalg.norm(Wpos, axis=1, keepdims=True), 1e-12)
    best_pos = (templates @ pos.T).max(axis=1)
    # Raw-weight match: stricter, exposes any off-bar (negative) structure.
    raw = Wa / np.linalg.norm(Wa, axis=1, keepdims=True)
    best_raw = (templates @ raw.T).max(axis=1)
    return {
        "recovered": int(np.sum(best_pos >= threshold)),
        "recovered_raw": int(np.sum(best_raw >= threshold)),
        "active_outputs": int(np.sum(active)),
        "mean_best_similarity": float(best_pos.mean()),
        "mean_best_similarity_raw": float(best_raw.mean()),
    }


class StereoDisparityData:
    """Two input streams; the right is the left shifted +/-1 pixel (thesis Ch.4.6.1)."""

    def __init__(self, width: int = 8, active: int = 6, prob: float = 0.5,
                 noise_sigma: float = 0.1, rng: np.random.Generator | int | None = None):
        self.width = width
        self.active = active
        self.prob = prob
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(rng)
        self.n_inputs = 2 * width

    def sample(self) -> np.ndarray:
        w, a = self.width, self.active
        left = np.zeros(w)
        start = (w - a) // 2
        left[start:start + a] = (self.rng.random(a) < self.prob).astype(float)
        shift = 1 if self.rng.random() < 0.5 else -1
        right = np.roll(left, shift)
        x = np.concatenate([left, right])
        return x + self.rng.normal(0.0, self.noise_sigma, size=x.shape)

    def sampler(self):
        return self.sample


class SignedBarsData:
    """Bars that appear as +1 or -1 (a bidirectional cause).

    A sample is the linear sum of independently chosen signed bar patterns. Used
    to probe the fragmentation cost of a non-negative output code (AbsTopK): a
    code with y >= 0 needs a separate dictionary atom for +bar and -bar, so it
    should require ~2x the outputs of a sign-preserving code.
    """

    def __init__(self, size: int = 8, prob: float = 1 / 8,
                 rng: np.random.Generator | int | None = None):
        self.size = size
        self.prob = prob
        self.rng = np.random.default_rng(rng)
        self.n_inputs = size * size
        self.n_bars = 2 * size

    def sample(self) -> np.ndarray:
        s = self.size
        grid = np.zeros((s, s))
        for i in range(s):                       # horizontal bars
            if self.rng.random() < self.prob:
                grid[i, :] += 1.0 if self.rng.random() < 0.5 else -1.0
        for j in range(s):                       # vertical bars
            if self.rng.random() < self.prob:
                grid[:, j] += 1.0 if self.rng.random() < 0.5 else -1.0
        return grid.ravel()

    def sampler(self):
        return self.sample

    def bar_templates(self) -> np.ndarray:
        return BarsData(size=self.size).bar_templates()


def evaluate_signed_bars_recovery(W: np.ndarray, size: int = 8,
                                  threshold: float = 0.9, active_norm: float = 0.1):
    """Score recovery of the 2*(2*size) signed half-features.

    Uses *signed* cosine against raw (not rectified) weight vectors, so +bar and
    -bar are distinct directions matched by different outputs.

    Returns a dict with:
      - recovered_signed: how many of the 2*(2*size) signed features are matched
        by some active output (signed cosine >= ``threshold``)
      - outputs_used: outputs whose weight-vector norm exceeds ``active_norm``
      - mean_best_similarity: average best-match signed cosine over all signed features
    """
    base = BarsData(size=size).bar_templates()             # (2*size, N) unit rows
    signed = np.vstack([base, -base])                      # (2*(2*size), N)
    norms = np.linalg.norm(W, axis=1)
    active = norms > active_norm
    Wn = W[active]
    if Wn.shape[0] == 0:
        return {"recovered_signed": 0, "outputs_used": 0, "mean_best_similarity": 0.0}
    Wn = Wn / np.linalg.norm(Wn, axis=1, keepdims=True)
    sims = signed @ Wn.T                                    # signed cosine
    best = sims.max(axis=1)
    return {
        "recovered_signed": int(np.sum(best >= threshold)),
        "outputs_used": int(np.sum(active)),
        "mean_best_similarity": float(best.mean()),
    }


def evaluate_signed_bars_recovery_abs(W: np.ndarray, size: int = 8,
                                      threshold: float = 0.9, active_norm: float = 0.1):
    """Recovery of the 2*size causes up to sign (absolute cosine).

    A cause (bar) is recovered if some active output's weight vector aligns with its
    template up to sign (|cosine| >= threshold). For a sign-preserving code one atom
    covers both the +bar and -bar of a cause, so all 2*size causes can be captured
    with 2*size atoms (vs 2*(2*size) signed features for the non-negative code).

    Returns a dict with:
      - recovered_causes: how many of the 2*size bar causes are matched up to sign
      - outputs_used: outputs whose weight-vector norm exceeds ``active_norm``
      - mean_best_abs_similarity: average best-match |cosine| over all causes
    """
    base = BarsData(size=size).bar_templates()
    norms = np.linalg.norm(W, axis=1)
    active = norms > active_norm
    Wn = W[active]
    if Wn.shape[0] == 0:
        return {"recovered_causes": 0, "outputs_used": 0, "mean_best_abs_similarity": 0.0}
    Wn = Wn / np.linalg.norm(Wn, axis=1, keepdims=True)
    sims = np.abs(base @ Wn.T)
    best = sims.max(axis=1)
    return {
        "recovered_causes": int(np.sum(best >= threshold)),
        "outputs_used": int(np.sum(active)),
        "mean_best_abs_similarity": float(best.mean()),
    }
