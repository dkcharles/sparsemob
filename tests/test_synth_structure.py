import numpy as np

from nfnet.synth import SyntheticFeatureData


def _corr_offdiag(fire, group_of, same):
    """Mean Pearson correlation between feature firings, within or across groups."""
    F = fire.astype(float)
    F = F - F.mean(axis=0, keepdims=True)
    std = np.sqrt((F * F).sum(axis=0))
    n = fire.shape[1]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            if std[i] > 0 and std[j] > 0 and (same == (group_of[i] == group_of[j])):
                vals.append((F[:, i] @ F[:, j]) / (std[i] * std[j]))
    return float(np.mean(vals)) if vals else 0.0


def test_independent_by_default():
    g = SyntheticFeatureData(d=16, n_features=32, rng=0)
    fire = g.sample_coeffs(4000) != 0
    within = _corr_offdiag(fire, g.group_of, same=True)
    assert abs(within) < 0.05            # no structure by default


def test_correlation_raises_within_group_cofiring():
    g = SyntheticFeatureData(d=16, n_features=32, n_groups=4, correlation=0.8, rng=0)
    fire = g.sample_coeffs(6000) != 0
    within = _corr_offdiag(fire, g.group_of, same=True)
    across = _corr_offdiag(fire, g.group_of, same=False)
    assert within > 0.2                  # same-group features co-fire
    assert within > across + 0.15        # and markedly more than cross-group


def test_correlation_preserves_marginal_rates():
    # Correlated firing must keep each feature's marginal firing probability ~ p_i.
    g = SyntheticFeatureData(d=16, n_features=32, n_groups=4, correlation=0.8, rng=1)
    fire = g.sample_coeffs(20000) != 0
    rate = fire.mean(axis=0)
    assert np.allclose(rate, g.p, atol=0.03)


def test_hierarchy_child_implies_parent():
    g = SyntheticFeatureData(d=16, n_features=31, hierarchy=True, branching=2, rng=0)
    fire = g.sample_coeffs(4000) != 0
    # For every non-root feature, every sample where the child fired must have the
    # parent firing too.
    for i in range(g.n_features):
        par = g.parent_of[i]
        if par >= 0:
            child_on = fire[:, i]
            assert np.all(fire[child_on, par])   # child on => parent on


def test_defaults_match_legacy_rng_stream():
    # With no structure, sample_coeffs must be identical to a fresh independent draw,
    # i.e. the RNG stream is untouched by the new machinery.
    g = SyntheticFeatureData(d=16, n_features=32, rng=7)
    c = g.sample_coeffs(50)
    g2 = SyntheticFeatureData(d=16, n_features=32, rng=7)
    fire = g2.rng.random((50, 32)) < g2.p
    mags = np.abs(g2.rng.normal(g2.mag_mean, g2.mag_std, size=(50, 32)))
    expected = fire * mags
    assert np.allclose(c, expected)
