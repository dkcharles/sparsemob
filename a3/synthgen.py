"""GPU (torch) SynthSAEBench-style generator at scale.

Mirrors nfnet.synth.SyntheticFeatureData but produces activation batches directly on a
torch device, so the d=768 / 16,384-feature regime is feasible on a GPU. Same design:
random unit-vector dictionary, Zipfian firing, folded-normal magnitudes, with optional
grouped correlation and parent-gated hierarchy.
"""

from __future__ import annotations

import torch


class TorchSyntheticFeatureData:
    def __init__(self, d: int = 768, n_features: int = 16384, p_min: float = 1e-4,
                 p_max: float = 0.1, zipf_exponent: float = 0.5, mag_mean: float = 4.5,
                 mag_std: float = 0.5, correlation: float = 0.0, n_groups: int = 256,
                 hierarchy: bool = False, branching: int = 8,
                 device: str = "cuda", seed: int = 0):
        self.device = torch.device(device)
        self.d = d
        self.n_features = n_features
        self.mag_mean = mag_mean
        self.mag_std = mag_std
        self.correlation = correlation
        self.n_groups = max(1, n_groups)
        self.hierarchy = hierarchy
        self.branching = max(1, branching)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        D = torch.randn(n_features, d, generator=self.gen, device=self.device)
        self.dictionary = D / D.norm(dim=1, keepdim=True)

        ranks = torch.arange(1, n_features + 1, device=self.device, dtype=torch.float32)
        self.p = (p_max * ranks.pow(-zipf_exponent)).clamp(p_min, p_max)

        self.group_of = torch.arange(n_features, device=self.device) % self.n_groups
        self.parent_of = torch.full((n_features,), -1, dtype=torch.long, device=self.device)
        if n_features > 1:
            idx = torch.arange(1, n_features, device=self.device)
            self.parent_of[1:] = (idx - 1) // self.branching

    def _fire(self, batch: int) -> torch.Tensor:
        u = torch.rand(batch, self.n_features, generator=self.gen, device=self.device)
        if self.correlation > 0.0:
            shared = torch.rand(batch, self.n_groups, generator=self.gen,
                                device=self.device)[:, self.group_of]
            use_shared = torch.rand(batch, self.n_features, generator=self.gen,
                                    device=self.device) < self.correlation
            u = torch.where(use_shared, shared, u)
        fire = u < self.p
        if self.hierarchy:
            # Topological (index) order; chunk to bound memory. Parents have lower index.
            for i in range(1, self.n_features):
                par = self.parent_of[i]
                if par >= 0:
                    fire[:, i] &= fire[:, par]
        return fire

    def sample_coeffs(self, batch: int) -> torch.Tensor:
        fire = self._fire(batch).float()
        z = torch.randn(batch, self.n_features, generator=self.gen, device=self.device)
        mags = (self.mag_mean + self.mag_std * z).abs()      # folded normal |N(mean,std)|
        return fire * mags

    def sample_batch(self, batch: int) -> torch.Tensor:
        return self.sample_coeffs(batch) @ self.dictionary
