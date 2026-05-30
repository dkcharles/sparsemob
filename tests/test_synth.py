import numpy as np

from nfnet.synth import SyntheticFeatureData, mcc_recovery


def test_batch_shape_and_continuous():
    g = SyntheticFeatureData(d=32, n_features=64, rng=0)
    X = g.sample_batch(50)
    assert X.shape == (50, 32)
    # continuous (not binary): more than a couple of distinct magnitudes
    assert np.unique(np.round(np.abs(X[X != 0]), 3)).size > 5
    assert g.dictionary.shape == (64, 32)


def test_nonnegative_default_signed_option():
    # default coefficients are non-negative; signed=True introduces negative coeffs.
    g_pos = SyntheticFeatureData(d=16, n_features=32, signed=False, rng=1)
    g_sig = SyntheticFeatureData(d=16, n_features=32, signed=True, rng=1)
    # With non-negative coeffs and non-negative checks via reconstruction coefficients:
    cpos = g_pos.sample_coeffs(200)
    csig = g_sig.sample_coeffs(200)
    assert cpos.min() >= 0.0
    assert csig.min() < 0.0


def test_sparsity_is_sparse():
    g = SyntheticFeatureData(d=32, n_features=128, rng=2)
    c = g.sample_coeffs(500)
    avg_active = (c != 0).sum(axis=1).mean()
    assert 1.0 < avg_active < 128 * 0.5   # genuinely sparse


def test_mcc_perfect_recovery():
    g = SyntheticFeatureData(d=32, n_features=48, rng=3)
    res = mcc_recovery(g.dictionary, g.dictionary)
    assert res["mcc"] > 0.999
    assert res["recovered"] == 48
    assert res["active_outputs"] == 48


def test_mcc_sign_insensitive():
    g = SyntheticFeatureData(d=32, n_features=48, rng=4)
    res = mcc_recovery(-g.dictionary, g.dictionary)
    assert res["mcc"] > 0.999          # abs cosine ignores the global sign flip


def test_mcc_partial():
    g = SyntheticFeatureData(d=32, n_features=48, rng=5)
    W = g.dictionary[:24].copy()       # only half the atoms present
    res = mcc_recovery(W, g.dictionary)
    assert res["active_outputs"] == 24
    assert res["recovered"] == 24      # the 24 present are matched at |cos|=1
