"""Scientific-target objectives for the autonomous loop.

Each objective scores a trained network per seed, decides pass/fail, and
aggregates across seeds into an ObjectiveResult. The agent minimises
primary_score; seed_pass_fraction (reliability across seeds) is the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nfnet.data import (
    evaluate_bars_recovery,
    evaluate_signed_bars_recovery,
    evaluate_signed_bars_recovery_abs,
)
from nfnet.synth import mcc_recovery


@dataclass
class ObjectiveResult:
    primary_score: float
    passed: bool
    margin: float
    seed_pass_fraction: float
    diagnostics: dict[str, float] = field(default_factory=dict)


class MOBObjective:
    """Minimum Overcomplete Basis: recover all causes with exactly n_causes active."""

    name = "mob"

    def __init__(self, size: int = 8):
        self.size = size
        self.n_causes = 2 * size

    def per_seed(self, W: np.ndarray) -> dict:
        r = evaluate_bars_recovery(W, size=self.size)
        recovered = r["recovered"]
        active = r["active_outputs"]
        score = (self.n_causes - recovered) + abs(active - self.n_causes)
        return {"recovered": recovered, "active": active, "score": score,
                "mean_best_similarity": r["mean_best_similarity"]}

    def passed(self, m: dict) -> bool:
        return m["recovered"] == self.n_causes and m["active"] == self.n_causes

    def aggregate(self, metrics: list[dict]) -> ObjectiveResult:
        if not metrics:
            raise ValueError("aggregate requires at least one seed's metrics")
        scores = np.array([m["score"] for m in metrics], dtype=float)
        passes = np.array([self.passed(m) for m in metrics], dtype=float)
        return ObjectiveResult(
            primary_score=float(scores.mean()),
            passed=bool(passes.all()),
            margin=float(scores.max()),          # worst-case distance to target
            seed_pass_fraction=float(passes.mean()),
            diagnostics={
                "mean_active": float(np.mean([m["active"] for m in metrics])),
                "mean_recovered": float(np.mean([m["recovered"] for m in metrics])),
            },
        )


class SignedBarsObjective:
    """Non-negativity fragmentation: recover all 2*(2*size) signed features.

    Note: primary_score = (n_signed - recovered_signed) + alpha * outputs_used,
    so a passing run (recovered_signed == n_signed) has a non-zero floor of
    alpha * outputs_used; the alpha term is a sparsity regulariser that, among
    passing runs, prefers fewer active outputs. seed_pass_fraction (not
    primary_score) is the primary success gate; primary_score is only a tiebreak
    among runs with equal pass fraction.
    """

    name = "signed_bars"

    def __init__(self, size: int = 8, alpha: float = 0.05):
        self.size = size
        self.alpha = alpha                       # penalty per output used
        self.n_signed = 2 * (2 * size)

    def per_seed(self, W: np.ndarray) -> dict:
        r = evaluate_signed_bars_recovery(W, size=self.size)
        recovered = r["recovered_signed"]
        used = r["outputs_used"]
        score = (self.n_signed - recovered) + self.alpha * used
        return {"recovered_signed": recovered, "outputs_used": used, "score": score,
                "mean_best_similarity": r["mean_best_similarity"]}

    def passed(self, m: dict) -> bool:
        return m["recovered_signed"] == self.n_signed

    def aggregate(self, metrics: list[dict]) -> ObjectiveResult:
        if not metrics:
            raise ValueError("aggregate requires at least one seed's metrics")
        scores = np.array([m["score"] for m in metrics], dtype=float)
        passes = np.array([self.passed(m) for m in metrics], dtype=float)
        return ObjectiveResult(
            primary_score=float(scores.mean()),
            passed=bool(passes.all()),
            margin=float(scores.max()),          # worst-case distance to target
            seed_pass_fraction=float(passes.mean()),
            diagnostics={
                "mean_recovered_signed": float(np.mean([m["recovered_signed"] for m in metrics])),
                "mean_outputs_used": float(np.mean([m["outputs_used"] for m in metrics])),
            },
        )


class SignedAbsObjective:
    """Sign-preserving recovery: capture all 2*size causes up to sign (abs cosine)."""

    name = "signed_bars_signed"

    def __init__(self, size: int = 8, alpha: float = 0.05):
        self.size = size
        self.alpha = alpha
        self.n_causes = 2 * size

    def per_seed(self, W: np.ndarray) -> dict:
        r = evaluate_signed_bars_recovery_abs(W, size=self.size)
        recovered = r["recovered_causes"]
        used = r["outputs_used"]
        score = (self.n_causes - recovered) + self.alpha * used
        return {"recovered_causes": recovered, "outputs_used": used, "score": score,
                "mean_best_abs_similarity": r["mean_best_abs_similarity"]}

    def passed(self, m: dict) -> bool:
        return m["recovered_causes"] == self.n_causes

    def aggregate(self, metrics: list[dict]) -> ObjectiveResult:
        if not metrics:
            raise ValueError("aggregate requires at least one seed's metrics")
        scores = np.array([m["score"] for m in metrics], dtype=float)
        passes = np.array([self.passed(m) for m in metrics], dtype=float)
        return ObjectiveResult(
            primary_score=float(scores.mean()),
            passed=bool(passes.all()),
            margin=float(scores.max()),          # worst-case distance to target
            seed_pass_fraction=float(passes.mean()),
            diagnostics={
                "mean_recovered_causes": float(np.mean([m["recovered_causes"] for m in metrics])),
                "mean_outputs_used": float(np.mean([m["outputs_used"] for m in metrics])),
            },
        )


class SynthObjective:
    """MCC-based recovery on SynthSAEBench-style synthetic features.

    Over-provision the coder (n_outputs > n_features) and let noise prune the active
    set toward n_features while keeping a high Mean Correlation Coefficient (MCC,
    Hungarian |cos|) against the ground-truth dictionary. ``per_seed`` takes the
    trained weights AND the seed's ground-truth dictionary.
    """

    name = "synth"

    def __init__(self, n_features: int, mcc_pass: float = 0.8,
                 active_tol: float = 0.1, alpha: float = 0.5):
        self.n_features = n_features
        self.mcc_pass = mcc_pass
        self.active_tol = active_tol
        self.alpha = alpha

    def per_seed(self, W: np.ndarray, dictionary: np.ndarray) -> dict:
        r = mcc_recovery(W, dictionary)
        score = (1.0 - r["mcc"]) + self.alpha * abs(r["active_outputs"] - self.n_features) / max(self.n_features, 1)
        return {"mcc": r["mcc"], "active": r["active_outputs"],
                "recovered": r["recovered"], "score": score}

    def passed(self, m: dict) -> bool:
        return (m["mcc"] >= self.mcc_pass
                and abs(m["active"] - self.n_features) <= self.active_tol * self.n_features)

    def aggregate(self, metrics: list[dict]) -> ObjectiveResult:
        if not metrics:
            raise ValueError("aggregate requires at least one seed's metrics")
        scores = np.array([m["score"] for m in metrics], dtype=float)
        passes = np.array([self.passed(m) for m in metrics], dtype=float)
        return ObjectiveResult(
            primary_score=float(scores.mean()),
            passed=bool(passes.all()),
            margin=float(scores.max()),          # worst-case distance to target
            seed_pass_fraction=float(passes.mean()),
            diagnostics={
                "mean_mcc": float(np.mean([m["mcc"] for m in metrics])),
                "mean_active": float(np.mean([m["active"] for m in metrics])),
                "mean_recovered": float(np.mean([m["recovered"] for m in metrics])),
            },
        )
