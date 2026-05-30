"""GPU (PyTorch) port of the negative-feedback coder, noise rules, and MCC metric.

This mirrors the NumPy implementation in ``nfnet`` (network.py, noise.py, synth.py) so
that the same experiments can run at SynthSAEBench scale (d=768, ~16k features) on a
GPU. The minibatched update is the vectorised negative-feedback rule:

    A = X W^T ; Y = f(A) (+ noise) ; E = X - Y W ; W += eta * (Y^T E) / B

Only PyTorch is required (no sae_lens). Everything runs on whatever ``device`` the
weights live on.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- #
# Non-linearities (element-wise), matching nfnet.nonlinearities                #
# --------------------------------------------------------------------------- #
def soft_threshold(tau: float = 1.0, lam: float = 4.0):
    """Positive soft threshold  y = log(1 + exp(lam (a - tau))) / lam."""
    def f(a: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(lam * (a - tau)) / lam
    return f


def soft_shrink(tau: float = 1.0, lam: float = 4.0):
    """Sign-preserving soft threshold  y = sign(a) log(1 + exp(lam(|a| - tau))) / lam."""
    def f(a: torch.Tensor) -> torch.Tensor:
        return torch.sign(a) * torch.nn.functional.softplus(lam * (a.abs() - tau)) / lam
    return f


# --------------------------------------------------------------------------- #
# Noise rules. Each is a callable returning a (B, M) noise tensor; stateful     #
# rules expose observe(Y) / observe_weights(W), mirroring the NumPy injectors.  #
# --------------------------------------------------------------------------- #
class UniformNoise:
    name = "uniform"

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, Y: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(Y) * self.sigma


class AdaptiveNoise:
    """Activity-normalised: sigma_i^2 proportional to the output's activity EMA."""

    name = "adaptive"

    def __init__(self, sigma: float = 0.1, beta: float = 0.99, eps: float = 1e-8):
        self.sigma = sigma
        self.beta = beta
        self.eps = eps
        self.activity: torch.Tensor | None = None

    def observe(self, Y: torch.Tensor) -> None:
        batch_activity = (Y * Y).mean(dim=0)
        if self.activity is None:
            self.activity = batch_activity
        else:
            self.activity = self.beta * self.activity + (1.0 - self.beta) * batch_activity

    def __call__(self, Y: torch.Tensor) -> torch.Tensor:
        if self.activity is None:
            return torch.randn_like(Y) * self.sigma
        scale = torch.sqrt((self.activity + self.eps) / (self.activity.mean() + self.eps))
        return torch.randn_like(Y) * (self.sigma * scale)


class RedundancyNoise:
    """Activation-correlation redundancy: sigma_i = sigma * max_j |corr(y_i, y_j)|."""

    name = "redundancy"

    def __init__(self, sigma: float = 0.1, beta: float = 0.9, eps: float = 1e-8):
        self.sigma = sigma
        self.beta = beta
        self.eps = eps
        self.redundancy: torch.Tensor | None = None

    def observe(self, Y: torch.Tensor) -> None:
        if Y.shape[0] < 2:
            return
        Yc = Y - Y.mean(dim=0, keepdim=True)
        std = torch.sqrt((Yc * Yc).sum(dim=0))
        denom = torch.outer(std, std)
        corr = torch.where(denom > self.eps, (Yc.t() @ Yc) / denom,
                           torch.zeros_like(denom))
        corr.fill_diagonal_(0.0)
        r = corr.abs().max(dim=1).values
        if self.redundancy is None:
            self.redundancy = r
        else:
            self.redundancy = self.beta * self.redundancy + (1.0 - self.beta) * r

    def __call__(self, Y: torch.Tensor) -> torch.Tensor:
        if self.redundancy is None:
            return torch.randn_like(Y) * self.sigma
        return torch.randn_like(Y) * (self.sigma * self.redundancy)


class WeightRedundancyNoise:
    """Weight-direction redundancy: sigma_i = sigma * max_j |cos(w_i, w_j)|."""

    name = "weight_redundancy"

    def __init__(self, sigma: float = 0.1, beta: float = 0.9, eps: float = 1e-8,
                 active_norm: float = 0.1):
        self.sigma = sigma
        self.beta = beta
        self.eps = eps
        self.active_norm = active_norm
        self.redundancy: torch.Tensor | None = None

    def observe_weights(self, W: torch.Tensor) -> None:
        norms = W.norm(dim=1, keepdim=True)
        Wn = W / norms.clamp_min(self.eps)
        C = (Wn @ Wn.t()).abs()
        C.fill_diagonal_(0.0)
        r = C.max(dim=1).values
        inactive = norms.squeeze(1) < self.active_norm
        r = torch.where(inactive, torch.zeros_like(r), r)
        if self.redundancy is None:
            self.redundancy = r
        else:
            self.redundancy = self.beta * self.redundancy + (1.0 - self.beta) * r

    def __call__(self, Y: torch.Tensor) -> torch.Tensor:
        if self.redundancy is None:
            return torch.randn_like(Y) * self.sigma
        return torch.randn_like(Y) * (self.sigma * self.redundancy)


# --------------------------------------------------------------------------- #
# The negative-feedback coder                                                 #
# --------------------------------------------------------------------------- #
class TorchNegativeFeedbackCoder:
    """Minibatched negative-feedback coder on a torch device (CPU or CUDA)."""

    def __init__(self, n_inputs: int, n_outputs: int, nonlinearity=None,
                 noise=None, weight_init: float = 1e-2, device: str = "cpu",
                 seed: int | None = None):
        self.device = torch.device(device)
        self.f = nonlinearity if nonlinearity is not None else (lambda a: a)
        self.noise = noise
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        self._gen = g
        self.W = (torch.rand(n_outputs, n_inputs, generator=g, device=self.device) * 2 - 1) * weight_init

    def train_step(self, X: torch.Tensor, eta: float) -> torch.Tensor:
        A = X @ self.W.t()
        Y = self.f(A)
        if self.noise is not None:
            if hasattr(self.noise, "observe"):
                self.noise.observe(Y)
            if hasattr(self.noise, "observe_weights"):
                self.noise.observe_weights(self.W)
            Y = Y + self.noise(Y)
        E = X - Y @ self.W
        self.W += eta * (Y.t() @ E) / X.shape[0]
        return Y

    def train(self, sampler_batch, n_steps: int, batch_size: int, eta0: float = 0.05):
        """``sampler_batch(B) -> (B, n_inputs)`` tensor on this device. Linear anneal."""
        for step in range(n_steps):
            eta = eta0 * (1.0 - step / max(n_steps, 1))
            X = sampler_batch(batch_size)
            self.train_step(X, eta)
        return self


# --------------------------------------------------------------------------- #
# MCC recovery metric (Hungarian matching on |cos|), matching nfnet.synth      #
# --------------------------------------------------------------------------- #
def mcc_recovery(W: torch.Tensor, dictionary: torch.Tensor, threshold: float = 0.5,
                 active_norm: float = 0.1) -> dict:
    """Mean Correlation Coefficient recovery. Hungarian matching is done on CPU."""
    from scipy.optimize import linear_sum_assignment

    norms = W.norm(dim=1)
    active = norms > active_norm
    Wn = W[active]
    if Wn.shape[0] == 0:
        return {"mcc": 0.0, "recovered": 0, "active_outputs": 0}
    Wn = Wn / Wn.norm(dim=1, keepdim=True)
    Dn = dictionary / dictionary.norm(dim=1, keepdim=True)
    sim = (Dn @ Wn.t()).abs().cpu().numpy()          # (n_features, n_active)
    rows, cols = linear_sum_assignment(-sim)
    matched = sim[rows, cols]
    return {
        "mcc": float(matched.mean()),
        "recovered": int((matched >= threshold).sum()),
        "active_outputs": int(active.sum().item()),
    }
