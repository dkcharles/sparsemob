"""Tests for the torch scaled generator (a3.synthgen). Run in the A3 venv."""

from __future__ import annotations

import torch

from a3.synthgen import TorchSyntheticFeatureData


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_shapes_and_dictionary_unit_norm():
    g = TorchSyntheticFeatureData(d=32, n_features=64, device=_dev(), seed=0)
    X = g.sample_batch(50)
    assert X.shape == (50, 32)
    assert g.dictionary.shape == (64, 32)
    assert torch.allclose(g.dictionary.norm(dim=1), torch.ones(64, device=g.device), atol=1e-5)


def test_sparse_and_continuous():
    g = TorchSyntheticFeatureData(d=32, n_features=128, device=_dev(), seed=1)
    c = g.sample_coeffs(500)
    avg_active = (c != 0).float().sum(dim=1).mean().item()
    assert 1.0 < avg_active < 128 * 0.5
    nz = c[c != 0]
    assert nz.unique().numel() > 5            # continuous magnitudes


def test_correlation_raises_within_group_cofiring():
    g = TorchSyntheticFeatureData(d=16, n_features=32, n_groups=4, correlation=0.8,
                                  device=_dev(), seed=0)
    fire = (g.sample_coeffs(4000) != 0).float().cpu()
    go = g.group_of.cpu()
    F = fire - fire.mean(0, keepdim=True)
    std = (F * F).sum(0).sqrt()
    within, across = [], []
    for i in range(32):
        for j in range(i + 1, 32):
            if std[i] > 0 and std[j] > 0:
                c = float((F[:, i] @ F[:, j]) / (std[i] * std[j]))
                (within if go[i] == go[j] else across).append(c)
    assert sum(within) / len(within) > sum(across) / len(across) + 0.1


def test_hierarchy_child_implies_parent():
    g = TorchSyntheticFeatureData(d=16, n_features=31, hierarchy=True, branching=2,
                                  device=_dev(), seed=0)
    fire = (g.sample_coeffs(2000) != 0).cpu()
    par = g.parent_of.cpu()
    for i in range(31):
        if par[i] >= 0:
            child_on = fire[:, i]
            assert bool(fire[child_on, par[i]].all())
