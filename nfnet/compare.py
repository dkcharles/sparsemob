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

    Matches min(rows) pairs; 1.0 iff the dictionaries span the same atoms.
    """
    A, B = _unit(W1), _unit(W2)
    S = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())
