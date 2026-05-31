# nfnet/compare.py
"""Metrics for comparing two learned dictionaries / width-control outcomes."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

ACTIVE_FLOOR = 0.1   # weight-norm threshold for "active" output (swept in the floor study)


def active_count(W: np.ndarray, floor: float = ACTIVE_FLOOR) -> int:
    return int((np.linalg.norm(W, axis=1) > floor).sum())


def _unit(W):
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.maximum(n, 1e-12)


def matched_atom_similarity(W1: np.ndarray, W2: np.ndarray) -> float:
    """Mean |cos| of a Hungarian matching between the rows of W1 and W2.

    Matches min(#rows) pairs and averages over matched pairs only, so it measures
    the quality of the matched atoms, not coverage: a small dictionary that aligns
    well with a subset of a larger one scores high while omitting many atoms, and
    duplicate or inactive rows further complicate interpretation. Report coverage
    (e.g. fraction of W2 atoms matched above a threshold) alongside this value;
    it reaches 1.0 when every matched pair is collinear, which requires equal atom
    counts and a perfect pairing but does not by itself certify identical spans.
    """
    A, B = _unit(W1), _unit(W2)
    S = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())
