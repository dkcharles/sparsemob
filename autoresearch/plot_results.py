"""Generate figures for the MOB experiment from the trial ledger.

Reads `autoresearch/ledger/experiments.jsonl` and writes PNGs to
`docs/results/figures/`. Also trains one network at the kept baseline config to
render a Hinton map of the recovered basis. Re-run after new batches:

    python -m autoresearch.plot_results
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
LEDGER = os.path.join(_HERE, "ledger", "experiments.jsonl")
FIGDIR = os.path.join(_HERE, "..", "docs", "results", "figures")


def load_rows():
    rows = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig_sigma_curve(rows):
    """Width-selection vs noise at the canonical 100k-step / 5-seed condition."""
    pts = []
    for r in rows:
        c, res = r["config"], r["result"]
        if r["experiment"] == "mob" and c["n_steps"] == 100000 and len(c["seeds"]) == 5:
            pts.append((c["sigma"], res["seed_pass_fraction"],
                        res["diagnostics"]["mean_active"]))
    pts = sorted(set(pts))
    sig = [p[0] for p in pts]
    pf = [p[1] for p in pts]
    ma = [p[2] for p in pts]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    l_active, = ax1.plot(sig, ma, "o-", color="tab:blue", label="mean active outputs")
    l_true = ax1.axhline(16, ls="--", color="grey", lw=1, label="true #causes (16)")
    ax1.set_xlabel("noise σ")
    ax1.set_ylabel("mean active outputs", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    l_pf, = ax2.plot(sig, pf, "s-", color="tab:red", label="seed pass fraction")
    ax2.set_ylabel("seed pass fraction", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_title("MOB width selection vs noise\n(24 outputs, 100k steps, 5 seeds)")
    ax1.legend(handles=[l_active, l_true, l_pf], loc="center right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "mob_sigma_curve.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_efficiency(rows):
    """Two-sided operating window in (n_steps, σ) space, coloured by reliability."""
    xs, ys, cs, over = [], [], [], []
    for r in rows:
        c, res = r["config"], r["result"]
        if r["experiment"] == "mob" and len(c["seeds"]) == 5:
            xs.append(c["n_steps"])
            ys.append(c["sigma"])
            cs.append(res["seed_pass_fraction"])
            over.append(res["diagnostics"]["mean_recovered"] < 15.5)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0, vmax=1, s=110,
                    edgecolor="k", zorder=2)
    ox = [x for x, o in zip(xs, over) if o]
    oy = [y for y, o in zip(ys, over) if o]
    if ox:
        ax.scatter(ox, oy, marker="x", color="red", s=130, zorder=3,
                   label="over-pruned (lost causes)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training steps (log scale)")
    ax.set_ylabel("noise σ")
    ax.set_title("Efficiency frontier: noise × training budget (5 seeds)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("seed pass fraction")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "mob_efficiency_frontier.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_baseline_hinton():
    """Hinton map of the kept efficiency-optimum config (σ=0.30, 25k steps)."""
    from nfnet import BarsData, NegativeFeedbackNet, linear_anneal
    from nfnet import nonlinearities as nl, noise as ns
    from nfnet.viz import save_hinton

    data = BarsData(size=8, prob=1 / 8, rng=0)
    net = NegativeFeedbackNet(
        64, n_outputs=24, nonlinearity=nl.soft_threshold(tau=1.0, lam=4.0),
        noise=ns.uniform_gaussian(sigma=0.30), weight_init=1e-3, rng=0,
    )
    net.train(data.sampler(), 25000, eta_schedule=linear_anneal(0.05))
    path = os.path.join(FIGDIR, "mob_baseline_hinton.png")
    save_hinton(net.W, path, grid_shape=(8, 8),
                title="MOB baseline σ=0.30, 25k steps (24 outputs → 16 bars)")
    return path


def fig_signed_fragmentation(rows):
    """Recovery vs n_outputs for signed bars: saturates at 2x the cause count."""
    pts = []
    for r in rows:
        c, res = r["config"], r["result"]
        if (r["experiment"] == "signed_bars" and c["sigma"] == 0.0
                and len(c["seeds"]) == 5):
            pts.append((c["n_outputs"], res["diagnostics"]["mean_recovered_signed"],
                        res["diagnostics"]["mean_outputs_used"]))
    pts = sorted(set(pts))
    no = [p[0] for p in pts]
    rec = [p[1] for p in pts]
    used = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(no, rec, "o-", color="tab:green", label="recovered signed features")
    ax.plot(no, used, "s--", color="tab:gray", alpha=0.7, label="active outputs used")
    ax.axhline(32, ls=":", color="tab:red", lw=1, label="all signed features (32)")
    ax.axvline(32, ls=":", color="tab:blue", lw=1, label="2 × #causes (32)")
    ax.set_xlabel("n_outputs")
    ax.set_ylabel("count")
    ax.set_title("Signed-bars fragmentation: recovery saturates at 2× causes\n"
                 "(σ=0, 100k steps, 5 seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "signed_fragmentation.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _train_signed(n_outputs, sigma, seed=0, n_steps=100000):
    """Train one signed-bars network and return its weight matrix W."""
    from nfnet.data import SignedBarsData
    from nfnet.network import NegativeFeedbackNet, linear_anneal
    from nfnet import nonlinearities as nl, noise as ns

    data = SignedBarsData(size=8, prob=1 / 8, rng=seed)
    net = NegativeFeedbackNet(
        64, n_outputs=n_outputs, nonlinearity=nl.soft_threshold(tau=1.0, lam=4.0),
        noise=ns.uniform_gaussian(sigma=sigma) if sigma > 0 else None,
        weight_init=1e-3, rng=seed,
    )
    net.train(data.sampler(), n_steps, eta_schedule=linear_anneal(0.05))
    return net.W


def fig_signed_hinton():
    """Hinton map of the 32-output signed solution: +bars (white) and -bars (black)."""
    from nfnet.viz import save_hinton

    W = _train_signed(32, 0.0)
    path = os.path.join(FIGDIR, "signed_bars_hinton.png")
    save_hinton(W, path, grid_shape=(8, 8),
                title="Signed bars, 32 outputs: +bars (white) and −bars (black)")
    return path


def fig_signed_hinton_variants():
    """Hinton maps below, at, and above the 32-atom basis size."""
    from nfnet.viz import save_hinton

    specs = [
        (16, 0.0, "signed_bars_hinton_n16.png",
         "16 outputs (under-complete): ~16 of 32 signed features"),
        (24, 0.0, "signed_bars_hinton_n24.png",
         "24 outputs (under-complete): ~24 of 32 signed features"),
        (40, 0.0, "signed_bars_hinton_n40.png",
         "40 outputs, no noise: 32 features + 8 unpruned spares"),
        (48, 0.15, "signed_bars_hinton_n48_noise.png",
         "48 outputs + noise (σ=0.15): pruned to the 32-atom MOB"),
    ]
    paths = []
    for n, sigma, fname, title in specs:
        W = _train_signed(n, sigma)
        p = os.path.join(FIGDIR, fname)
        save_hinton(W, p, grid_shape=(8, 8), title=title)
        paths.append(p)
    return paths


def fig_signed_comparison(rows):
    """Non-negative vs sign-preserving: fraction of target recovered per n_outputs."""
    nn, sp = [], []
    for r in rows:
        c, res = r["config"], r["result"]
        if c["sigma"] == 0.0 and len(c["seeds"]) == 5:
            if r["experiment"] == "signed_bars":
                nn.append((c["n_outputs"], res["diagnostics"]["mean_recovered_signed"] / 32.0))
            elif r["experiment"] == "signed_bars_signed":
                sp.append((c["n_outputs"], res["diagnostics"]["mean_recovered_causes"] / 16.0))
    nn, sp = sorted(set(nn)), sorted(set(sp))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot([p[0] for p in nn], [p[1] for p in nn], "o-", color="tab:purple",
            label="non-negative code (needs 32 atoms)")
    ax.plot([p[0] for p in sp], [p[1] for p in sp], "s-", color="tab:green",
            label="sign-preserving code (needs 16 atoms)")
    ax.axvline(16, ls=":", color="tab:green", lw=1)
    ax.axvline(32, ls=":", color="tab:purple", lw=1)
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_xlabel("n_outputs (atoms)")
    ax.set_ylabel("fraction of target recovered")
    ax.set_title("Sign-preserving halves the atoms needed\n(signed bars, σ=0, 5 seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "signed_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_signed_prune_frontier(rows):
    """Noise pruning of the over-complete signed net (48 outputs) toward the 32-atom MOB."""
    pts = []
    for r in rows:
        c, res = r["config"], r["result"]
        if (r["experiment"] == "signed_bars" and c["n_outputs"] == 48
                and c["n_steps"] == 100000 and len(c["seeds"]) == 5):
            pts.append((c["sigma"], res["diagnostics"]["mean_outputs_used"],
                        res["diagnostics"]["mean_recovered_signed"]))
    pts = sorted(set(pts))
    sig = [p[0] for p in pts]
    used = [p[1] for p in pts]
    rec = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sig, used, "o-", color="tab:blue", label="active outputs used")
    ax.plot(sig, rec, "s-", color="tab:green", label="recovered signed features")
    ax.axhline(32, ls="--", color="grey", lw=1, label="MOB target (32)")
    ax.set_xlabel("noise σ")
    ax.set_ylabel("count")
    ax.set_title("Signed MOB pruning: noise prunes 48 outputs toward 32\n"
                 "(n_outputs=48, 100k steps, 5 seeds)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "signed_prune_frontier.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_signed_signed_hinton():
    """Hinton map of the 16-atom sign-preserving solution (one atom per cause)."""
    from nfnet.data import SignedBarsData
    from nfnet.network import NegativeFeedbackNet, linear_anneal
    from nfnet import nonlinearities as nl
    from nfnet.viz import save_hinton

    data = SignedBarsData(size=8, prob=1 / 8, rng=0)
    net = NegativeFeedbackNet(
        64, n_outputs=16, nonlinearity=nl.soft_shrink(tau=1.0, lam=4.0),
        noise=None, weight_init=1e-3, rng=0,
    )
    net.train(data.sampler(), 100000, eta_schedule=linear_anneal(0.05))
    path = os.path.join(FIGDIR, "signed_signed_hinton.png")
    save_hinton(net.W, path, grid_shape=(8, 8),
                title="Sign-preserving, 16 outputs: all 16 causes (one atom each)")
    return path


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rows = load_rows()
    print("loaded", len(rows), "ledger rows")
    print("saved", fig_sigma_curve(rows))
    print("saved", fig_efficiency(rows))
    print("saved", fig_baseline_hinton())
    print("saved", fig_signed_fragmentation(rows))
    print("saved", fig_signed_hinton())
    for p in fig_signed_hinton_variants():
        print("saved", p)
    print("saved", fig_signed_comparison(rows))
    print("saved", fig_signed_prune_frontier(rows))
    print("saved", fig_signed_signed_hinton())


if __name__ == "__main__":
    main()
