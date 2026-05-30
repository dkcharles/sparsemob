"""Tests for the five-axis noise-vs-L2 identity harness."""
from autoresearch.run_identity import identity_trial


def test_identity_trial_returns_five_axis_record_small():
    rec = identity_trial(experiment="mob", strengths=[0.0, 0.02], seeds=[0, 1],
                         n_steps=500, n_outputs=24)
    for k in ("noise", "l2"):
        assert k in rec and "active_mean" in rec[k] and "active_std" in rec[k]
    assert "matched_atom_cos" in rec
    assert "trajectory" in rec


def test_five_axis_lengths_aligned():
    """All per-strength lists must have the same length as strengths."""
    strengths = [0.0, 0.01, 0.05]
    rec = identity_trial(experiment="mob", strengths=strengths, seeds=[0],
                         n_steps=300, n_outputs=20)
    n = len(strengths)
    for mech in ("noise", "l2"):
        for stat in ("active_mean", "active_std", "recovery_mean", "recovery_std"):
            assert len(rec[mech][stat]) == n, f"{mech}/{stat} length mismatch"
    assert len(rec["matched_atom_cos"]) == n
    assert rec["strengths"] == list(strengths)


def test_trajectory_structure():
    """Trajectory must have steps/noise/l2 sub-lists of equal length."""
    rec = identity_trial(experiment="mob", strengths=[0.0, 0.02], seeds=[0],
                         n_steps=400, n_outputs=20)
    traj = rec["trajectory"]
    assert "steps" in traj and "noise" in traj and "l2" in traj
    assert len(traj["steps"]) == len(traj["noise"]) == len(traj["l2"])
    assert len(traj["steps"]) > 0


def test_recovery_and_active_are_numeric():
    """Check that values are plain Python numbers (JSON-serialisable)."""
    rec = identity_trial(experiment="mob", strengths=[0.02], seeds=[0],
                         n_steps=300, n_outputs=20)
    for mech in ("noise", "l2"):
        for stat in ("active_mean", "active_std", "recovery_mean", "recovery_std"):
            v = rec[mech][stat][0]
            assert isinstance(v, (int, float)), f"{mech}/{stat} not numeric: {type(v)}"
    assert isinstance(rec["matched_atom_cos"][0], (int, float))
