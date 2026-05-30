# tests/test_width_control.py
import numpy as np
from nfnet.network import NegativeFeedbackNet
from nfnet.nonlinearities import identity


def test_weight_decay_shrinks_weights_each_step():
    # With no input drive (zeros) and weight_decay>0, weights must shrink toward 0.
    net = NegativeFeedbackNet(4, 3, nonlinearity=identity, weight_decay=0.1,
                              weight_init=0.5, rng=0)
    before = np.linalg.norm(net.W)
    X = np.zeros((8, 4))
    net.train_step_batch(X, eta=0.1, step=0)
    after = np.linalg.norm(net.W)
    assert after < before


def test_weight_decay_zero_is_unchanged_behaviour():
    # weight_decay=0 must reproduce the exact same update as before (regression guard).
    rng = 0
    X = np.random.default_rng(1).normal(size=(8, 4))
    a = NegativeFeedbackNet(4, 3, nonlinearity=identity, weight_decay=0.0, rng=rng)
    b = NegativeFeedbackNet(4, 3, nonlinearity=identity, rng=rng)
    a.train_step_batch(X, eta=0.1); b.train_step_batch(X, eta=0.1)
    assert np.allclose(a.W, b.W)


def test_act_l1_soft_shrinks_outputs():
    # act_l1 applies the L1 prox sign(y)*max(|y|-lam,0) to outputs before learning.
    net = NegativeFeedbackNet(4, 3, nonlinearity=identity, act_l1=0.2, weight_init=0.0, rng=0)
    Y = net._apply_selection(np.array([[1.0, 0.1, 1.0]]))
    # 0.1 magnitude shrinks to 0 (below 0.2); 1.0 shrinks to 0.8.
    assert np.allclose(Y, [[0.8, 0.0, 0.8]])


def test_topk_keeps_only_k_largest_per_sample():
    net = NegativeFeedbackNet(4, 4, nonlinearity=identity, topk=2, rng=0)
    Y = net._apply_selection(np.array([[0.1, 0.9, 0.5, 0.2]]))
    assert np.allclose(Y, [[0.0, 0.9, 0.5, 0.0]])


def test_feedback_off_uses_input_as_residual():
    # With feedback off, E = X (no reconstruction subtraction), so the update is eta*Y^T X.
    rng = 0
    X = np.random.default_rng(2).normal(size=(8, 4))
    net = NegativeFeedbackNet(4, 3, nonlinearity=identity, feedback=False,
                              weight_init=0.1, rng=rng)
    W0 = net.W.copy()
    net.train_step_batch(X, eta=0.1, step=0)
    A = X @ W0.T
    expected = W0 + 0.1 * (A.T @ X) / X.shape[0]   # Y=A (identity), E=X
    assert np.allclose(net.W, expected)
