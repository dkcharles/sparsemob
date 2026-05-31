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


def test_weight_decay_is_simultaneous_update():
    # L2 decay must use the pre-update weights: W += eta*(H - wd*W),
    # not the sequential (1-eta*wd)(W+eta*H) which carries an extra -eta^2*wd*H term.
    X = np.random.default_rng(3).normal(size=(8, 4))
    net = NegativeFeedbackNet(4, 3, nonlinearity=identity, weight_decay=0.2,
                              weight_init=0.1, rng=0)
    W0 = net.W.copy()
    net.train_step_batch(X, eta=0.1, step=0)
    Y = X @ W0.T                       # identity nonlinearity
    E = X - Y @ W0
    H = (Y.T @ E) / X.shape[0]
    expected = W0 + 0.1 * (H - 0.2 * W0)
    assert np.allclose(net.W, expected)


def test_weight_redundancy_ignores_inactive_columns():
    from nfnet.noise import WeightRedundancyNoise
    W = np.array([[1.0, 0.0, 0.0],     # active
                  [1e-3, 0.0, 0.0],    # inactive, aligned with row 0
                  [0.0, 1.0, 0.0]])    # active, orthogonal to row 0
    nz = WeightRedundancyNoise(sigma=0.1, active_norm=0.1)
    nz.observe_weights(W)
    r = nz.redundancy
    assert r[0] == 0.0   # only aligned partner is inactive -> not redundant
    assert r[1] == 0.0   # inactive output
    assert r[2] == 0.0   # orthogonal to the only other active output


def test_bars_metric_reports_raw_and_positive():
    from nfnet.data import evaluate_bars_recovery
    W = np.random.default_rng(0).normal(size=(24, 64))
    r = evaluate_bars_recovery(W)
    assert {"recovered", "recovered_raw", "active_outputs",
            "mean_best_similarity", "mean_best_similarity_raw"} <= set(r)
    # Positive-part clipping can only raise alignment with the non-negative templates.
    assert r["mean_best_similarity"] >= r["mean_best_similarity_raw"] - 1e-9


def test_topk_zero_returns_zero_code():
    net = NegativeFeedbackNet(4, 4, nonlinearity=identity, topk=0, rng=0)
    Y = net._apply_selection(np.array([[0.1, 0.9, 0.5, 0.2]]))
    assert np.allclose(Y, 0.0)


def test_online_adaptive_noise_updates_state():
    # Stateful noise must adapt in the online train() path, not only in train_batched().
    from nfnet.noise import AdaptiveNoise
    nz = AdaptiveNoise(sigma=0.1)
    net = NegativeFeedbackNet(6, 4, nonlinearity=identity, noise=nz, weight_init=0.1, rng=0)
    assert nz.activity is None
    rng = np.random.default_rng(0)
    for _ in range(5):
        net.train_step(rng.normal(size=6), eta=0.01)
    assert nz.activity is not None                 # observe() was called from the online path
    assert nz.activity.shape == (4,)


def test_forward_applies_l1_and_topk_selection():
    # forward() must return the same code path that training uses (selection applied).
    net = NegativeFeedbackNet(4, 4, nonlinearity=identity, topk=2, rng=0)
    net.W = np.eye(4)
    _, y = net.forward(np.array([0.1, 0.9, 0.5, 0.2]))
    assert np.count_nonzero(y) == 2                 # top-k keeps only 2 of 4
    assert set(np.flatnonzero(y)) == {1, 2}         # the two largest magnitudes

    net2 = NegativeFeedbackNet(3, 3, nonlinearity=identity, act_l1=0.2, rng=0)
    net2.W = np.eye(3)
    _, y2 = net2.forward(np.array([0.5, 0.1, -0.5]))
    # soft-threshold shrinks magnitudes by act_l1 and zeros sub-threshold ones.
    assert np.allclose(y2, [0.3, 0.0, -0.3])


def test_plot_selector_rejects_feedback_off_and_l2_rows():
    # The figure _canonical() predicate must exclude the revision ablation runs.
    # paper/ is not part of the public code mirror, so skip there.
    import pytest
    mf = pytest.importorskip("paper.make_figures")
    _canonical = mf._canonical
    canonical = {"width_control": "noise", "feedback": True, "act_l1": 0.0, "topk": None}
    assert _canonical(canonical)
    assert not _canonical({**canonical, "width_control": "l2"})
    assert not _canonical({**canonical, "feedback": False})
    assert not _canonical({**canonical, "act_l1": 0.1})
    assert not _canonical({**canonical, "topk": 16})


def test_constructor_validates_parameters():
    import pytest
    with pytest.raises(ValueError):
        NegativeFeedbackNet(4, 4, topk=-1, rng=0)
    with pytest.raises(ValueError):
        NegativeFeedbackNet(4, 4, topk=5, rng=0)            # > n_outputs
    with pytest.raises(ValueError):
        NegativeFeedbackNet(4, 4, weight_decay=-0.1, rng=0)
    with pytest.raises(ValueError):
        NegativeFeedbackNet(4, 4, act_l1=-0.1, rng=0)
    with pytest.raises(ValueError):
        NegativeFeedbackNet(0, 4, rng=0)
    # boundary values are allowed
    NegativeFeedbackNet(4, 4, topk=0, rng=0)
    NegativeFeedbackNet(4, 4, topk=4, rng=0)
