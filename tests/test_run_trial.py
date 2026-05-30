import json

from autoresearch.run_trial import TrialConfig, run_trial, decide_keep, append_ledger
from autoresearch.objectives import ObjectiveResult


def test_run_trial_mob_smoke():
    cfg = TrialConfig(
        experiment="mob", tau=1.0, lam=4.0, eta0=0.05, sigma=0.1,
        n_outputs=24, weight_init=1e-3, n_steps=2000, seeds=[0],
    )
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    # Structure only; 2000 steps is too short to assert a pass.
    assert 0.0 <= res.seed_pass_fraction <= 1.0
    assert "mean_active" in res.diagnostics


def test_run_trial_signed_smoke():
    cfg = TrialConfig(
        experiment="signed_bars", tau=1.0, lam=4.0, eta0=0.05, sigma=0.0,
        n_outputs=32, weight_init=1e-3, n_steps=2000, seeds=[0],
    )
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert 0.0 <= res.seed_pass_fraction <= 1.0
    assert "mean_outputs_used" in res.diagnostics


def test_run_trial_signed_signed_smoke():
    cfg = TrialConfig(
        experiment="signed_bars_signed", tau=1.0, lam=4.0, eta0=0.05, sigma=0.0,
        n_outputs=16, weight_init=1e-3, n_steps=2000, seeds=[0],
    )
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_recovered_causes" in res.diagnostics


def test_run_trial_synth_smoke():
    cfg = TrialConfig(
        experiment="synth", d=32, n_features=32, n_outputs=48, sigma=0.0,
        tau=1.0, lam=4.0, eta0=0.02, weight_init=1e-2, n_steps=200,
        batch_size=64, seeds=[0],
    )
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_mcc" in res.diagnostics
    assert 0.0 <= res.diagnostics["mean_mcc"] <= 1.0


def test_run_trial_synth_adaptive_smoke():
    cfg = TrialConfig(experiment="synth", d=32, n_features=32, n_outputs=48,
                      sigma=0.1, noise_mode="adaptive", eta0=0.02, weight_init=1e-2,
                      n_steps=200, batch_size=64, seeds=[0])
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_mcc" in res.diagnostics


def test_run_trial_synth_controller_smoke():
    cfg = TrialConfig(experiment="synth", d=32, n_features=32, n_outputs=48,
                      sigma=0.1, noise_mode="controller", eta0=0.02, weight_init=1e-2,
                      n_steps=200, batch_size=64, seeds=[0])
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_mcc" in res.diagnostics


def test_run_trial_synth_redundancy_smoke():
    cfg = TrialConfig(experiment="synth", d=32, n_features=32, n_outputs=48,
                      sigma=0.1, noise_mode="redundancy", eta0=0.02, weight_init=1e-2,
                      n_steps=200, batch_size=64, seeds=[0])
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_mcc" in res.diagnostics


def _res(spf, primary):
    return ObjectiveResult(primary_score=primary, passed=spf == 1.0,
                           margin=primary, seed_pass_fraction=spf, diagnostics={})


def test_decide_keep_first_trial_always_kept():
    cfg = TrialConfig(experiment="mob", n_steps=1000)
    assert decide_keep(_res(0.6, 2.0), cfg, baseline=None) is True


def test_decide_keep_prefers_higher_pass_fraction():
    cfg = TrialConfig(experiment="mob", n_steps=1000)
    base = {"primary_score": 2.0, "seed_pass_fraction": 0.6, "n_steps": 1000}
    assert decide_keep(_res(0.8, 2.0), cfg, base) is True
    assert decide_keep(_res(0.4, 0.0), cfg, base) is False


def test_decide_keep_breaks_ties_on_primary_then_steps():
    base = {"primary_score": 1.0, "seed_pass_fraction": 0.8, "n_steps": 100_000}
    cfg = TrialConfig(experiment="mob", n_steps=100_000)
    assert decide_keep(_res(0.8, 0.5), cfg, base) is True            # lower primary
    cfg_fast = TrialConfig(experiment="mob", n_steps=50_000)
    assert decide_keep(_res(0.8, 1.0), cfg_fast, base) is True       # tie primary, fewer steps


def test_append_ledger_writes_rows_and_updates_baseline(tmp_path):
    cfg = TrialConfig(experiment="mob", n_steps=1000, seeds=[0])
    res = _res(1.0, 0.0)
    ledger_dir = tmp_path / "ledger"
    baseline_path = tmp_path / "baseline.json"
    kept = append_ledger(cfg, res, rationale="first trial", ledger_dir=str(ledger_dir),
                          baseline_path=str(baseline_path))
    assert kept is True
    lines = (ledger_dir / "experiments.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["kept"] is True
    assert row["rationale"] == "first trial"
    assert row["result"]["seed_pass_fraction"] == 1.0
    saved = json.loads(baseline_path.read_text())
    assert saved["result"]["seed_pass_fraction"] == 1.0
    assert (ledger_dir / "log.md").exists()


def test_append_ledger_discard_keeps_old_baseline(tmp_path):
    cfg = TrialConfig(experiment="mob", n_steps=1000, seeds=[0])
    ledger_dir = tmp_path / "ledger"
    baseline_path = tmp_path / "baseline.json"
    # First trial is kept and writes the baseline.
    append_ledger(cfg, _res(1.0, 0.0), rationale="first",
                  ledger_dir=str(ledger_dir), baseline_path=str(baseline_path))
    # A worse trial (lower pass fraction) must be discarded, baseline unchanged.
    kept = append_ledger(cfg, _res(0.4, 5.0), rationale="worse",
                         ledger_dir=str(ledger_dir), baseline_path=str(baseline_path))
    assert kept is False
    saved = json.loads(baseline_path.read_text())
    assert saved["result"]["seed_pass_fraction"] == 1.0
    lines = (ledger_dir / "experiments.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_decide_keep_rejects_no_improvement():
    base = {"primary_score": 1.0, "seed_pass_fraction": 0.8, "n_steps": 1000, "margin": 1.0}
    cfg = TrialConfig(experiment="mob", n_steps=1000)
    # Identical spf, primary, steps -> no improvement -> not kept.
    assert decide_keep(_res(0.8, 1.0), cfg, base) is False


def test_decide_keep_rejects_worst_case_regression():
    from autoresearch.objectives import ObjectiveResult
    base = {"primary_score": 1.0, "seed_pass_fraction": 0.8, "n_steps": 1000, "margin": 1.0}
    cfg = TrialConfig(experiment="mob", n_steps=1000)
    # Higher pass fraction but worst-case (margin) regressed past tolerance -> rejected.
    res = ObjectiveResult(primary_score=0.5, passed=False, margin=3.0,
                          seed_pass_fraction=0.9, diagnostics={})
    assert decide_keep(res, cfg, base) is False


def test_run_trial_synth_weight_redundancy_smoke():
    cfg = TrialConfig(experiment="synth", d=32, n_features=32, n_outputs=48,
                      sigma=0.1, noise_mode="weight_redundancy", eta0=0.02,
                      weight_init=1e-2, n_steps=200, batch_size=64, seeds=[0])
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_mcc" in res.diagnostics


def test_run_trial_synth_correlation_hierarchy_smoke():
    cfg = TrialConfig(experiment="synth", d=32, n_features=32, n_outputs=48,
                      sigma=0.1, noise_mode="uniform", eta0=0.02, weight_init=1e-2,
                      n_steps=200, batch_size=64, seeds=[0],
                      correlation=0.6, n_groups=4, hierarchy=True, branching=3)
    res = run_trial(cfg)
    assert isinstance(res, ObjectiveResult)
    assert "mean_mcc" in res.diagnostics


def test_width_control_l2_runs_and_prunes_on_mob():
    cfg = TrialConfig(experiment="mob", sigma=0.0, width_control="l2",
                      weight_decay=0.02, n_outputs=24, n_steps=2000, seeds=[0])
    res = run_trial(cfg)
    assert "mean_active" in res.diagnostics
