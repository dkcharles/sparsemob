"""SynthSAEBench-style synthetic feature data and MCC recovery metric.

A downscaled, dependency-light reimplementation of the generative design in
SynthSAEBench (Chanin et al., 2026): a ground-truth dictionary of random unit-vector
features in R^d combined with Zipfian sparse firing and folded-normal magnitudes.
Used to test whether the noise->MOB and sign-preserving results transfer from binary
bars to continuous, superposed features.

Optional firing structure (also in SynthSAEBench) is supported: ``correlation`` makes
features fire in correlated groups, and ``hierarchy`` gates a child feature so it can
only fire when its parent fires. Both change only the firing process; the ground-truth
dictionary directions are unaffected.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class SyntheticFeatureData:
    """Continuous, superposed multiple-cause data with a known ground-truth dictionary.

    Each sample x = sum_i c_i d_i, where d_i are random unit vectors in R^d, feature i
    fires with Zipfian probability p_i, and the (non-negative by default) coefficient
    c_i is a folded-normal magnitude. With ``signed=True`` coefficients may be negative
    (a bidirectional cause).

    Optional firing structure (off by default, so the independent-firing behaviour and
    RNG stream are preserved):
      - ``correlation`` in [0, 1]: features are partitioned into ``n_groups`` groups and
        co-fire within a group with strength ``correlation`` (marginal rates preserved).
      - ``hierarchy``: features form an index-ordered ``branching``-ary tree; a child can
        fire only when its parent fires (parent-gated co-activation).
    """

    def __init__(self, d: int = 128, n_features: int = 512, p_min: float = 1e-3,
                 p_max: float = 0.2, zipf_exponent: float = 0.5, mag_mean: float = 4.5,
                 mag_std: float = 0.5, signed: bool = False, correlation: float = 0.0,
                 n_groups: int = 16, hierarchy: bool = False, branching: int = 4,
                 rng: np.random.Generator | int | None = None):
        self.d = d
        self.n_inputs = d
        self.n_features = n_features
        self.signed = signed
        self.mag_mean = mag_mean
        self.mag_std = mag_std
        self.rng = np.random.default_rng(rng)

        # Ground-truth dictionary: random unit vectors (superposition via random dirs).
        D = self.rng.standard_normal((n_features, d))
        self.dictionary = D / np.linalg.norm(D, axis=1, keepdims=True)

        # Zipfian firing probabilities over feature rank, clipped to [p_min, p_max].
        ranks = np.arange(1, n_features + 1, dtype=float)
        self.p = np.clip(p_max * ranks ** (-zipf_exponent), p_min, p_max)

        # Optional firing-structure parameters (no RNG draws here, so the sampling
        # stream is unchanged when structure is off).
        self.correlation = correlation
        self.n_groups = max(1, n_groups)
        self.hierarchy = hierarchy
        self.branching = max(1, branching)
        # Group assignment, interleaved so group is decoupled from Zipfian rank.
        self.group_of = np.arange(n_features) % self.n_groups
        # Parent tree by index: feature 0 is the root; parent_of[i] = (i-1)//branching.
        self.parent_of = np.full(n_features, -1, dtype=int)
        if n_features > 1:
            idx = np.arange(1, n_features)
            self.parent_of[1:] = (idx - 1) // self.branching

    def _fire(self, batch: int) -> np.ndarray:
        """Per-feature firing mask, shape (batch, n_features), with optional structure."""
        u = self.rng.random((batch, self.n_features))
        if self.correlation > 0.0:
            shared = self.rng.random((batch, self.n_groups))[:, self.group_of]
            use_shared = self.rng.random((batch, self.n_features)) < self.correlation
            u = np.where(use_shared, shared, u)
        fire = u < self.p
        if self.hierarchy:
            for i in range(1, self.n_features):      # parents have lower index
                par = self.parent_of[i]
                if par >= 0:
                    fire[:, i] &= fire[:, par]       # child fires only if parent fired
        return fire

    def sample_coeffs(self, batch: int) -> np.ndarray:
        """Return the (batch, n_features) coefficient matrix (mostly zero)."""
        fire = self._fire(batch)
        mags = np.abs(self.rng.normal(self.mag_mean, self.mag_std,
                                      size=(batch, self.n_features)))
        coeffs = fire * mags
        if self.signed:
            signs = np.where(self.rng.random((batch, self.n_features)) < 0.5, 1.0, -1.0)
            coeffs = coeffs * signs
        return coeffs

    def sample_batch(self, batch: int) -> np.ndarray:
        """Return a (batch, d) batch of activation vectors."""
        return self.sample_coeffs(batch) @ self.dictionary

    def sample(self) -> np.ndarray:
        return self.sample_batch(1)[0]


def mcc_recovery(W: np.ndarray, dictionary: np.ndarray, threshold: float = 0.5,
                 active_norm: float = 0.1):
    """Mean Correlation Coefficient recovery (SynthSAEBench-style).

    Matches active learned atoms (rows of ``W``) to ground-truth feature directions
    (rows of ``dictionary``) by Hungarian matching on absolute cosine similarity.

    Returns a dict with:
      - mcc: mean |cos| over the matched ground-truth features
      - recovered: # matched features with |cos| >= ``threshold``
      - active_outputs: # rows of W with norm > ``active_norm``
    """
    norms = np.linalg.norm(W, axis=1)
    active = norms > active_norm
    Wn = W[active]
    if Wn.shape[0] == 0:
        return {"mcc": 0.0, "recovered": 0, "active_outputs": 0}
    Wn = Wn / np.linalg.norm(Wn, axis=1, keepdims=True)
    Dn = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)
    sim = np.abs(Dn @ Wn.T)                       # (n_features, n_active), abs cosine
    rows, cols = linear_sum_assignment(-sim)      # maximise total matched similarity
    matched = sim[rows, cols]
    return {
        "mcc": float(matched.mean()),
        "recovered": int(np.sum(matched >= threshold)),
        "active_outputs": int(np.sum(active)),
    }
