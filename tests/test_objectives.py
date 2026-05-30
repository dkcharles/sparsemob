import numpy as np

from nfnet.data import BarsData
from autoresearch.objectives import MOBObjective, ObjectiveResult


def _mob_weights(active_bars=16, dead=8, size=8):
    base = BarsData(size=size).bar_templates()             # (16, 64) unit rows
    rows = [base[i] for i in range(active_bars)]
    rows += [np.zeros(size * size) for _ in range(dead)]
    return np.array(rows)


def test_mob_perfect_run_scores_zero_and_passes():
    obj = MOBObjective(size=8)
    W = _mob_weights(active_bars=16, dead=8)
    m = obj.per_seed(W)
    assert m["recovered"] == 16
    assert m["active"] == 16
    assert m["score"] == 0
    assert obj.passed(m) is True


def test_mob_aggregate_reports_pass_fraction():
    obj = MOBObjective(size=8)
    good = obj.per_seed(_mob_weights(16, 8))
    res = obj.aggregate([good, good, good])
    assert isinstance(res, ObjectiveResult)
    assert res.seed_pass_fraction == 1.0
    assert res.primary_score == 0.0


def test_signed_objective_full_recovery_passes():
    from nfnet.data import SignedBarsData
    from autoresearch.objectives import SignedBarsObjective

    size = 8
    base = SignedBarsData(size=size).bar_templates()
    W = np.vstack([base, -base])                 # 32 signed atoms
    obj = SignedBarsObjective(size=size)
    m = obj.per_seed(W)
    assert m["recovered_signed"] == 32
    assert obj.passed(m) is True
    res = obj.aggregate([m, m])
    assert res.seed_pass_fraction == 1.0


def test_signed_objective_positive_only_fails():
    from nfnet.data import SignedBarsData
    from autoresearch.objectives import SignedBarsObjective

    size = 8
    base = SignedBarsData(size=size).bar_templates()
    obj = SignedBarsObjective(size=size)
    m = obj.per_seed(base)                        # only 16 positive atoms
    assert m["recovered_signed"] == 16
    assert obj.passed(m) is False


def test_signed_abs_objective_full_recovery_passes():
    from nfnet.data import SignedBarsData
    from autoresearch.objectives import SignedAbsObjective

    size = 8
    base = SignedBarsData(size=size).bar_templates()   # 16 atoms, one per cause
    obj = SignedAbsObjective(size=size)
    m = obj.per_seed(base)
    assert m["recovered_causes"] == 16
    assert obj.passed(m) is True
    res = obj.aggregate([m, m])
    assert res.seed_pass_fraction == 1.0


def test_signed_abs_objective_partial_fails():
    from nfnet.data import SignedBarsData
    from autoresearch.objectives import SignedAbsObjective

    size = 8
    base = SignedBarsData(size=size).bar_templates()
    obj = SignedAbsObjective(size=size)
    m = obj.per_seed(base[:8])   # only 8 atoms -> 8 of 16 causes
    assert m["recovered_causes"] == 8
    assert obj.passed(m) is False
