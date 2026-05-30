"""Hinton-map visualisation of weight vectors (thesis figure style).

This is the only module that depends on matplotlib. Each output's weight vector
is reshaped to the input grid and drawn as a Hinton diagram: one square per
weight, area proportional to magnitude, white = positive, black = negative, on a
grey background. Outputs are tiled into a near-square panel grid.

:class:`HintonSnapshot` is a Callback, so it plugs into training to save PNGs
periodically. A future live viewer would implement the same Callback interface
against ``TrainingState.weights`` instead of writing files.
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")  # file output; no interactive backend needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

from .callbacks import Callback, TrainingState

_GREY = "#9a9a9a"


def _hinton_panel(ax, vec: np.ndarray, grid_shape, vmax: float):
    gh, gw = grid_shape
    m = vec.reshape(gh, gw)
    ax.set_facecolor(_GREY)
    ax.set_xlim(-0.5, gw - 0.5)
    ax.set_ylim(-0.5, gh - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for r in range(gh):
        for c in range(gw):
            w = m[r, c]
            if w == 0 or vmax == 0:
                continue
            size = math.sqrt(min(abs(w) / vmax, 1.0))
            color = "white" if w > 0 else "black"
            ax.add_patch(Rectangle(
                (c - size / 2, r - size / 2), size, size,
                facecolor=color, edgecolor="none",
            ))


def hinton_figure(W: np.ndarray, grid_shape=(8, 8), title: str | None = None,
                  n_cols: int | None = None, panel_size: float = 1.1):
    """Build a matplotlib Figure of Hinton panels, one per output neuron.

    ``panel_size`` controls inches per panel cell; increase for print-quality output.
    """
    n_out = W.shape[0]
    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(n_out)))
    n_rows = int(math.ceil(n_out / n_cols))
    vmax = float(np.abs(W).max()) or 1.0

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * panel_size, n_rows * panel_size),
                             squeeze=False)
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols][k % n_cols]
        if k < n_out:
            _hinton_panel(ax, W[k], grid_shape, vmax)
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.92 if title else 0.98,
                        left=0.02, right=0.98, bottom=0.02)
    return fig


def save_hinton(W: np.ndarray, path: str, grid_shape=(8, 8), title: str | None = None,
                n_cols: int | None = None, dpi: int = 120, panel_size: float = 1.1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig = hinton_figure(W, grid_shape=grid_shape, title=title, n_cols=n_cols,
                        panel_size=panel_size)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


class HintonSnapshot(Callback):
    """Save a Hinton-map PNG during/after training.

    Files are written to ``out_dir/{prefix}_{step}.png``. Wrap in
    :class:`nfnet.callbacks.Periodic` to control snapshot frequency.
    """

    def __init__(self, out_dir: str, prefix: str, grid_shape=(8, 8),
                 n_cols: int | None = None):
        self.out_dir = out_dir
        self.prefix = prefix
        self.grid_shape = grid_shape
        self.n_cols = n_cols

    def _save(self, state: TrainingState, tag: str):
        path = os.path.join(self.out_dir, f"{self.prefix}_{tag}.png")
        save_hinton(state.weights, path, grid_shape=self.grid_shape,
                    title=f"{self.prefix}  step {state.step}", n_cols=self.n_cols)
        return path

    def on_step(self, state: TrainingState) -> None:
        self._save(state, f"{state.step:07d}")

    def on_end(self, state: TrainingState) -> None:
        print("  saved", self._save(state, "final"))
