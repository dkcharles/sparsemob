import numpy as np

from nfnet.noise import WeightRedundancyNoise
from nfnet.network import NegativeFeedbackNet, linear_anneal
from nfnet import nonlinearities as nl


def test_weight_redundancy_high_noise_on_duplicate_rows():
    # Rows 0 and 1 are near-duplicate directions; row 2 is orthogonal; row 3 is zero.
    W = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.99, 0.01, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
    nz = WeightRedundancyNoise(sigma=0.1, beta=0.0)   # latest-only
    nz.observe_weights(W)
    rng = np.random.default_rng(0)
    sd = np.array([nz(0, 4, rng) for _ in range(4000)]).std(axis=0)
    assert sd[0] > sd[2] and sd[1] > sd[2]            # duplicates noisier than the unique row
    assert sd[2] < 0.05                                # unique (orthogonal) row: low noise
    assert sd[3] < 0.05                                # inactive (zero) row: low noise


def test_weight_redundancy_uniform_before_observe():
    nz = WeightRedundancyNoise(sigma=0.1)
    rng = np.random.default_rng(1)
    sd = np.array([nz(0, 5, rng) for _ in range(4000)]).std(axis=0)
    assert np.all(sd > 0.0)
    assert sd.max() / max(sd.min(), 1e-9) < 1.5


def test_weight_redundancy_no_nan_on_zero_rows():
    W = np.zeros((4, 6))
    W[0, 0] = 1.0
    nz = WeightRedundancyNoise(sigma=0.1, beta=0.0)
    nz.observe_weights(W)
    rng = np.random.default_rng(2)
    assert np.all(np.isfinite(nz(0, 4, rng)))


def test_weight_redundancy_observed_in_training():
    from nfnet.synth import SyntheticFeatureData
    g = SyntheticFeatureData(d=16, n_features=16, rng=0)
    nz = WeightRedundancyNoise(sigma=0.1)
    net = NegativeFeedbackNet(16, 24, nonlinearity=nl.soft_threshold(0.5, 4.0),
                              noise=nz, weight_init=1e-2, rng=0)
    net.train_batched(g.sample_batch, n_steps=200, batch_size=64,
                      eta_schedule=linear_anneal(0.02))
    assert np.all(np.isfinite(net.W))
    assert nz.redundancy is not None
