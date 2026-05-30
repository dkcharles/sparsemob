import numpy as np

from nfnet.noise import AdaptiveNoise, uniform_gaussian
from nfnet.network import NegativeFeedbackNet, linear_anneal
from nfnet import nonlinearities as nl


def test_adaptive_has_observe_and_scales_with_activity():
    rng = np.random.default_rng(0)
    nz = AdaptiveNoise(sigma=0.1, beta=0.5)
    # Output 0 is highly active, output 1 inactive, output 2 moderate.
    Y = np.array([[10.0, 0.0, 2.0],
                  [10.0, 0.0, 2.0]])
    nz.observe(Y)                 # builds the activity EMA
    draws = np.array([nz(step=10, n_outputs=3, rng=rng) for _ in range(4000)])
    sd = draws.std(axis=0)
    # Per-output noise SD tracks activity ordering: out0 > out2 > out1.
    assert sd[0] > sd[2] > sd[1]
    # The inactive output gets near-zero noise.
    assert sd[1] < 0.02


def test_adaptive_uniform_before_observation():
    # With no activity observed yet, all outputs share (approximately) one sigma.
    rng = np.random.default_rng(1)
    nz = AdaptiveNoise(sigma=0.1)
    draws = np.array([nz(step=0, n_outputs=4, rng=rng) for _ in range(4000)])
    sd = draws.std(axis=0)
    assert np.all(sd > 0.0)
    assert sd.max() / max(sd.min(), 1e-9) < 1.5     # roughly uniform at start


def test_observe_hook_called_in_batched_training():
    # A network with an AdaptiveNoise injector trains without error and the
    # injector's activity estimate becomes non-uniform after training.
    from nfnet.synth import SyntheticFeatureData
    g = SyntheticFeatureData(d=16, n_features=16, rng=0)
    nz = AdaptiveNoise(sigma=0.05)
    net = NegativeFeedbackNet(16, 24, nonlinearity=nl.soft_threshold(0.5, 4.0),
                              noise=nz, weight_init=1e-2, rng=0)
    net.train_batched(g.sample_batch, n_steps=200, batch_size=64,
                      eta_schedule=linear_anneal(0.02))
    assert np.all(np.isfinite(net.W))
    assert nz.activity is not None
    assert nz.activity.std() > 0.0      # activity differentiated across outputs


def test_uniform_injector_still_works_in_batched_path():
    # Backward-compat: a plain injector (no observe) trains fine in the batched path.
    from nfnet.synth import SyntheticFeatureData
    g = SyntheticFeatureData(d=16, n_features=16, rng=2)
    net = NegativeFeedbackNet(16, 24, nonlinearity=nl.soft_threshold(0.5, 4.0),
                              noise=uniform_gaussian(0.05), weight_init=1e-2, rng=2)
    net.train_batched(g.sample_batch, n_steps=100, batch_size=64,
                      eta_schedule=linear_anneal(0.02))
    assert np.all(np.isfinite(net.W))
