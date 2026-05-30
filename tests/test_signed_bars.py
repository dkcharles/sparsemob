import numpy as np

from nfnet.data import SignedBarsData, evaluate_signed_bars_recovery, evaluate_signed_bars_recovery_abs


def test_signed_sample_shape_and_signs():
    d = SignedBarsData(size=8, prob=0.5, rng=0)
    xs = np.array([d.sample() for _ in range(200)])
    assert xs.shape == (200, 64)
    # With signed bars and high prob, we should see both signs across samples.
    assert xs.min() < 0.0
    assert xs.max() > 0.0


def test_perfect_dictionary_recovers_all_signed_features():
    # Build a weight matrix equal to the 32 signed bar templates (+/- each bar).
    size = 8
    base = SignedBarsData(size=size).bar_templates()       # (16, 64) unit rows
    W = np.vstack([base, -base])                            # (32, 64)
    res = evaluate_signed_bars_recovery(W, size=size, threshold=0.9)
    assert res["recovered_signed"] == 32
    assert res["outputs_used"] == 32


def test_positive_only_dictionary_misses_negative_features():
    size = 8
    base = SignedBarsData(size=size).bar_templates()       # (16, 64)
    W = base.copy()                                        # only +bars
    res = evaluate_signed_bars_recovery(W, size=size, threshold=0.9)
    # Recovers the 16 positive directions, not the 16 negative ones.
    assert res["recovered_signed"] == 16


def test_all_inactive_returns_zeros():
    W = np.zeros((4, 64))
    res = evaluate_signed_bars_recovery(W)
    assert res == {"recovered_signed": 0, "outputs_used": 0, "mean_best_similarity": 0.0}


def test_abs_recovery_full_with_one_atom_per_cause():
    size = 8
    base = SignedBarsData(size=size).bar_templates()   # 16 unit rows, one per cause
    res = evaluate_signed_bars_recovery_abs(base, size=size, threshold=0.9)
    assert res["recovered_causes"] == 16
    assert res["outputs_used"] == 16


def test_abs_recovery_is_sign_insensitive():
    size = 8
    base = SignedBarsData(size=size).bar_templates()
    res = evaluate_signed_bars_recovery_abs(-base, size=size, threshold=0.9)  # negated atoms
    assert res["recovered_causes"] == 16   # |cosine| still matches
