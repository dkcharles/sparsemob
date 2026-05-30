import numpy as np

from nfnet.network import NegativeFeedbackNet, linear_anneal
from nfnet import nonlinearities as nl


def test_batch_update_matches_closed_form():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 8))
    net = NegativeFeedbackNet(8, 4, nonlinearity=nl.identity, rng=0)
    W0 = net.W.copy()
    eta = 0.01
    net.train_step_batch(X, eta)
    # Closed form for identity f, no noise: W += eta * (Y^T E)/B with Y=XW0^T, E=X-YW0.
    Y = X @ W0.T
    E = X - Y @ W0
    expected = W0 + eta * (Y.T @ E) / 5
    assert np.allclose(net.W, expected)


def test_batch_update_shapes_and_finite():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((32, 16))
    net = NegativeFeedbackNet(16, 24, nonlinearity=nl.soft_threshold(1.0, 4.0), rng=1)
    Y = net.train_step_batch(X, 0.01)
    assert Y.shape == (32, 24)
    assert net.W.shape == (24, 16)
    assert np.all(np.isfinite(net.W))


def test_train_batched_runs_on_synthetic():
    from nfnet.synth import SyntheticFeatureData, mcc_recovery
    g = SyntheticFeatureData(d=16, n_features=16, rng=0)
    net = NegativeFeedbackNet(16, 16, nonlinearity=nl.soft_threshold(0.5, 4.0),
                              weight_init=1e-2, rng=0)
    net.train_batched(g.sample_batch, n_steps=500, batch_size=64,
                      eta_schedule=linear_anneal(0.02))
    res = mcc_recovery(net.W, g.dictionary)
    assert 0.0 <= res["mcc"] <= 1.0
    assert np.all(np.isfinite(net.W))


def test_train_batched_with_noise_runs():
    from nfnet.synth import SyntheticFeatureData
    from nfnet import noise as ns
    g = SyntheticFeatureData(d=16, n_features=16, rng=2)
    net = NegativeFeedbackNet(16, 24, nonlinearity=nl.soft_threshold(0.5, 4.0),
                              noise=ns.uniform_gaussian(0.05), weight_init=1e-2, rng=2)
    net.train_batched(g.sample_batch, n_steps=300, batch_size=64,
                      eta_schedule=linear_anneal(0.02))
    assert np.all(np.isfinite(net.W))
