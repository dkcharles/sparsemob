import numpy as np

from nfnet.noise import RedundancyNoise
from nfnet.network import NegativeFeedbackNet, linear_anneal
from nfnet import nonlinearities as nl


def _batch_with_duplicate(n=400, rng=None):
    """Outputs 0 and 1 are duplicates (same signal); output 2 is independent."""
    rng = rng or np.random.default_rng(0)
    s = rng.normal(size=n)               # shared signal for cols 0,1
    u = rng.normal(size=n)               # independent signal for col 2
    Y = np.stack([s, s + 0.01 * rng.normal(size=n), u], axis=1)   # (n, 3)
    return Y


def test_redundancy_higher_noise_on_duplicates():
    rng = np.random.default_rng(0)
    nz = RedundancyNoise(sigma=0.1, beta=0.0)     # beta=0 -> use latest batch only
    nz.observe(_batch_with_duplicate(rng=rng))
    draws = np.array([nz(step=10, n_outputs=3, rng=rng) for _ in range(4000)])
    sd = draws.std(axis=0)
    # The two correlated (duplicate) outputs get more noise than the independent one.
    assert sd[0] > sd[2] and sd[1] > sd[2]
    assert sd[2] < 0.05                            # near-unique output: low noise


def test_redundancy_uniform_before_observation():
    rng = np.random.default_rng(1)
    nz = RedundancyNoise(sigma=0.1)
    draws = np.array([nz(0, 4, rng) for _ in range(4000)])
    sd = draws.std(axis=0)
    assert np.all(sd > 0.0)
    assert sd.max() / max(sd.min(), 1e-9) < 1.5    # roughly uniform at start


def test_observe_handles_zero_variance_columns():
    # A column that never varies (zero variance) must not produce NaN noise.
    nz = RedundancyNoise(sigma=0.1, beta=0.0)
    Y = np.zeros((50, 3))
    Y[:, 0] = np.linspace(-1, 1, 50)               # col 0 varies; cols 1,2 constant
    nz.observe(Y)
    rng = np.random.default_rng(2)
    draw = nz(0, 3, rng)
    assert np.all(np.isfinite(draw))


def test_redundancy_runs_in_batched_training():
    from nfnet.synth import SyntheticFeatureData
    g = SyntheticFeatureData(d=16, n_features=16, rng=0)
    nz = RedundancyNoise(sigma=0.1)
    net = NegativeFeedbackNet(16, 24, nonlinearity=nl.soft_threshold(0.5, 4.0),
                              noise=nz, weight_init=1e-2, rng=0)
    net.train_batched(g.sample_batch, n_steps=200, batch_size=64,
                      eta_schedule=linear_anneal(0.02))
    assert np.all(np.isfinite(net.W))
    assert nz.redundancy is not None
