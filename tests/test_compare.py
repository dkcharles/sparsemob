# tests/test_compare.py
import numpy as np
from nfnet.compare import active_count, matched_atom_similarity, ACTIVE_FLOOR


def test_active_count_uses_floor():
    W = np.array([[1.0, 0, 0], [1e-6, 0, 0], [0, 2.0, 0]])
    assert active_count(W) == 2                     # row 2 below ACTIVE_FLOOR
    assert active_count(W, floor=10.0) == 0


def test_matched_atom_similarity_identical_is_one():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(5, 8))
    perm = W[[3, 1, 4, 0, 2]]                        # same atoms, permuted
    assert matched_atom_similarity(W, perm) > 0.999


def test_matched_atom_similarity_orthogonal_is_low():
    W1 = np.eye(3, 6); W2 = np.eye(3, 6)[:, ::-1]
    assert matched_atom_similarity(W1, W2) < 0.1
