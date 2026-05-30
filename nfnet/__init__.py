"""nfnet -- negative feedback networks for multiple-cause coding.

A NumPy reimplementation of the unsupervised neural networks in
Darryl Charles' 1999 PhD thesis, "Unsupervised Artificial Neural Networks for
the Identification of Multiple Causes in Data".

The numerical core (network, nonlinearities, noise, data, callbacks) has no
UI/plotting dependency; visualisation lives in ``nfnet.viz``.
"""

from . import nonlinearities, noise
from .callbacks import Periodic, ProgressLogger, StateRecorder, TrainingState
from .data import BarsData, StereoDisparityData, evaluate_bars_recovery
from .network import NegativeFeedbackNet, constant_eta, linear_anneal

__all__ = [
    "NegativeFeedbackNet",
    "linear_anneal",
    "constant_eta",
    "BarsData",
    "StereoDisparityData",
    "evaluate_bars_recovery",
    "TrainingState",
    "Periodic",
    "ProgressLogger",
    "StateRecorder",
    "nonlinearities",
    "noise",
]
