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
