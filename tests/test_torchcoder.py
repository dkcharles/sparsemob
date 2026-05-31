"""Tests for the torch GPU port (a3.torchcoder). Run with the A3 venv:

    .venv-a3/Scripts/python.exe -m pytest tests/test_torchcoder.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from a3.torchcoder import (
    AdaptiveNoise, RedundancyNoise, TorchNegativeFeedbackCoder, UniformNoise,
    WeightRedundancyNoise, mcc_recovery, soft_shrink, soft_threshold,
)


def test_soft_threshold_matches_numpy():
    a = torch.linspace(-3, 3, 50)
    got = soft_threshold(1.0, 4.0)(a).numpy()
    exp = np.logaddexp(0.0, 4.0 * (a.numpy() - 1.0)) / 4.0
    assert np.allclose(got, exp, atol=1e-5)


def test_soft_shrink_odd_symmetric():
    f = soft_shrink(1.0, 4.0)
    a = torch.tensor([-3.0, -0.5, 0.5, 3.0])
    assert torch.allclose(f(-a), -f(a), atol=1e-6)


def test_train_step_matches_closed_form_identity():
    torch.manual_seed(0)
    X = torch.randn(5, 8)
    net = TorchNegativeFeedbackCoder(8, 4, nonlinearity=lambda a: a, weight_init=0.1, seed=0)
    W0 = net.W.clone()
    net.train_step(X, eta=0.01)
    Y = X @ W0.t()
    E = X - Y @ W0
    expected = W0 + 0.01 * (Y.t() @ E) / 5
    assert torch.allclose(net.W, expected, atol=1e-6)


def test_perfect_dictionary_gives_mcc_one():
    torch.manual_seed(0)
    D = torch.randn(48, 32)
    D = D / D.norm(dim=1, keepdim=True)
    res = mcc_recovery(D, D)
    assert res["mcc"] > 0.999
    assert res["recovered"] == 48
    assert res["active_outputs"] == 48


def test_mcc_sign_insensitive():
    torch.manual_seed(1)
    D = torch.randn(40, 32); D = D / D.norm(dim=1, keepdim=True)
    res = mcc_recovery(-D, D)
    assert res["mcc"] > 0.999


def test_adaptive_scales_with_activity():
    nz = AdaptiveNoise(sigma=0.1, beta=0.0)
    Y = torch.tensor([[10.0, 0.0, 2.0], [10.0, 0.0, 2.0]])
    nz.observe(Y)
    torch.manual_seed(0)
    draws = torch.stack([nz(Y) for _ in range(2000)])
    sd = draws.reshape(-1, 3).std(dim=0)
    assert sd[0] > sd[2] > sd[1]


def test_redundancy_flags_duplicates():
    nz = RedundancyNoise(sigma=0.1, beta=0.0)
    s = torch.randn(400)
    Y = torch.stack([s, s + 0.01 * torch.randn(400), torch.randn(400)], dim=1)
    nz.observe(Y)
    assert nz.redundancy[0] > nz.redundancy[2]
    assert nz.redundancy[1] > nz.redundancy[2]


def test_weight_redundancy_flags_duplicate_rows():
    nz = WeightRedundancyNoise(sigma=0.1, beta=0.0)
    W = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.99, 0.01, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
    nz.observe_weights(W)
    assert nz.redundancy[0] > nz.redundancy[2]
    assert nz.redundancy[3] < 0.05          # inactive row


def test_training_runs_and_is_finite():
    torch.manual_seed(0)
    D = torch.randn(16, 16); D = D / D.norm(dim=1, keepdim=True)

    def sampler(b):
        fire = (torch.rand(b, 16) < 0.2).float()
        return (fire * torch.randn(b, 16).abs()) @ D

    net = TorchNegativeFeedbackCoder(16, 24, nonlinearity=soft_threshold(0.5, 4.0),
                                     noise=UniformNoise(0.05), weight_init=1e-2, seed=0)
    net.train(sampler, n_steps=200, batch_size=64, eta0=0.02)
    assert torch.isfinite(net.W).all()
    res = mcc_recovery(net.W, D)
    assert 0.0 <= res["mcc"] <= 1.0


def test_torch_weight_decay_shrinks():
    import torch
    from a3.torchcoder import TorchNegativeFeedbackCoder, soft_threshold
    net = TorchNegativeFeedbackCoder(8, 6, nonlinearity=soft_threshold(1.0, 4.0),
                                     weight_decay=0.1, device="cpu", seed=0)
    before = net.W.norm().item()
    net.train_step(torch.zeros(4, 8), eta=0.1)
    assert net.W.norm().item() < before


def test_runs_on_cuda_if_available():
    if not torch.cuda.is_available():
        return
    D = torch.randn(32, 64, device="cuda"); D = D / D.norm(dim=1, keepdim=True)

    def sampler(b):
        fire = (torch.rand(b, 32, device="cuda") < 0.2).float()
        return (fire * torch.randn(b, 32, device="cuda").abs()) @ D

    net = TorchNegativeFeedbackCoder(64, 48, nonlinearity=soft_threshold(0.5, 4.0),
                                     noise=UniformNoise(0.05), weight_init=1e-2,
                                     device="cuda", seed=0)
    net.train(sampler, n_steps=100, batch_size=128, eta0=0.02)
    assert net.W.is_cuda
    assert torch.isfinite(net.W).all()


def test_torch_weight_decay_is_simultaneous():
    # delta = (Y^T E)/B - wd*W, applied as W += eta*delta (uses pre-update W).
    torch.manual_seed(0)
    X = torch.randn(6, 8)
    net = TorchNegativeFeedbackCoder(8, 4, nonlinearity=lambda a: a,
                                     weight_decay=0.2, weight_init=0.1, seed=0)
    W0 = net.W.clone()
    net.train_step(X, eta=0.1)
    Y = X @ W0.t()
    E = X - Y @ W0
    H = (Y.t() @ E) / X.shape[0]
    expected = W0 + 0.1 * (H - 0.2 * W0)
    assert torch.allclose(net.W, expected, atol=1e-6)


def test_torch_noise_is_seed_reproducible():
    # Two coders with the same seed must inject the identical noise stream,
    # independent of the global torch RNG state between them.
    def make():
        return TorchNegativeFeedbackCoder(8, 6, nonlinearity=lambda a: a,
                                          noise=UniformNoise(0.1), weight_init=0.0,
                                          seed=123)
    a = make()
    torch.manual_seed(999)               # perturb global RNG between constructions
    _ = torch.randn(100)
    b = make()
    X = torch.ones(4, 8)
    a.train_step(X, eta=0.0)             # eta=0 -> W unchanged, but noise drawn
    b.train_step(X, eta=0.0)
    # Draw a noise sample directly from each (generators are now in lock-step).
    na = a.noise(torch.zeros(3, 6))
    nb = b.noise(torch.zeros(3, 6))
    assert torch.allclose(na, nb)


def test_torch_noise_differs_across_seeds():
    a = TorchNegativeFeedbackCoder(8, 6, noise=UniformNoise(0.1), seed=1)
    b = TorchNegativeFeedbackCoder(8, 6, noise=UniformNoise(0.1), seed=2)
    na = a.noise(torch.zeros(4, 6))
    nb = b.noise(torch.zeros(4, 6))
    assert not torch.allclose(na, nb)


def test_torch_weight_redundancy_ignores_inactive_columns():
    nz = WeightRedundancyNoise(sigma=0.1, beta=0.0, active_norm=0.1)
    W = torch.tensor([[1.0, 0.0, 0.0],     # active
                      [1e-3, 0.0, 0.0],    # inactive, aligned with row 0
                      [0.0, 1.0, 0.0]])    # active, orthogonal to row 0
    nz.observe_weights(W)
    r = nz.redundancy
    assert r[0].item() == 0.0   # only aligned partner is inactive
    assert r[1].item() == 0.0   # inactive output
    assert r[2].item() == 0.0   # orthogonal to the only other active output
