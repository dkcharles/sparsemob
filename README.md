# sparsemob

**Noise selects a Minimum Overcomplete Basis.** This repository is a clean, runnable
implementation of a rectification-constrained, negative-feedback Hebbian coder and the
experiments that connect it to modern sparse autoencoders (SAEs) and the superposition
hypothesis.

The central object is a multiple-cause coder trained by a single local rule. When its
outputs are rectified and perturbed by additive noise, the network drives unused outputs
to zero and settles on a *Minimum Overcomplete Basis* (MOB): only as many outputs stay
active as there are underlying causes. The same construction is recognisably the ancestor
of today's SAE machinery (the soft-threshold output is a JumpReLU; positivity drives the
bidirectional feature-splitting that AbsTopK addresses; the noise penalty plays the role
of a width selector). The experiments here map both where this behaviour holds and where
it breaks down on realistic, superposed, skew-importance data.

## The mechanism

The whole model is one update rule (`nfnet/network.py`):

```
a = W x                  feedforward
y = f(a) (+ noise)       non-linearity, optional additive output noise
e = x - Wᵀy              feedback residual at the inputs
W += η · outer(y, e)     Hebbian learning on the residual
```

`f = identity` recovers the Oja subspace (PCA). A positive non-linearity recovers
individual causes. Adding output noise selects the MOB. The minibatched form
(`train_step_batch`) exposes `observe`/`observe_weights` hooks so that adaptive and
redundancy-based noise rules can watch the code as it trains.

## Layout

```
nfnet/                 the numerical core (pure NumPy, no UI dependency)
  network.py           NegativeFeedbackNet: train_step, train_step_batch, train
  nonlinearities.py    f(a): identity, rectify, soft_threshold (JumpReLU-style), ...
  noise.py             additive output noise: uniform, adaptive, redundancy, weight-redundancy
  data.py              BarsData, SignedBarsData + bar-recovery metrics
  synth.py             SynthSAEBench-style generator (Zipf firing, folded-normal
                       magnitudes, correlation + hierarchy) + MCC recovery metric
  callbacks.py         TrainingState + Callback protocol (matplotlib-free)
  viz.py               Hinton maps (the only plotting module)
autoresearch/          knob-only experiment harness (locked mechanism, scored objectives,
                       JSONL ledger); see autoresearch/program.md
experiments/           one runnable script per replicated bars experiment
a3/                    GPU (torch) port for benchmark-scale runs (d=768, 16,384 features)
tests/                 pytest suite (core is pure NumPy; GPU tests are CUDA-gated)
```

The numerical core has no UI or plotting dependency, so it stays portable: a live viewer
or a port to another language only needs to consume the `TrainingState` snapshot stream
from `nfnet/callbacks.py`.

## Install and run (core, CPU)

```bash
pip install -r requirements.txt

# Replicated bars experiments (Hinton-map PNGs are written to outputs/)
python -m experiments.exp01_bars       # 16 bars: linear vs soft-threshold
python -m experiments.exp02_bars_mob   # MOB: 24 outputs, with vs without noise
```

## The autoresearch harness

`autoresearch/` wraps the coder in a knob-only harness: the mechanism is locked, a config
sets the knobs, trials run across seeds, objectives are scored, and results are appended
to a JSONL ledger. One trial:

```bash
# MOB width selection
python -m autoresearch.run_trial --experiment mob --sigma 0.1 --n-outputs 24

# Synthetic SAE-style features (continuous, superposed, skew-importance)
python -m autoresearch.run_trial --experiment synth --noise-mode uniform --sigma 0.1 \
    --d 64 --n-features 128 --n-outputs 192

# Correlated / hierarchical firing structure
python -m autoresearch.run_trial --experiment synth --correlation 0.8 --n-groups 16
python -m autoresearch.run_trial --experiment synth --hierarchy --branching 4
```

Noise rules (`--noise-mode`): `uniform`, `adaptive` (activity-normalised), `controller`
(closed-loop count target), `redundancy` (activation-correlation), `weight_redundancy`
(weight-direction cosine). Baseline configs are in `autoresearch/configs/`.

## Benchmark-scale experiments (optional GPU)

`a3/` ports the coder, the four noise rules, and the generator to PyTorch so the
experiments run at the scale of a contemporary synthetic interpretability benchmark
(activation dimension 768, 16,384 ground-truth features, 24,576 outputs). Install the GPU
extra into an isolated environment (see `requirements-gpu.txt`), then:

```bash
# Count-selection sweep at scale (uniform noise)
bash a3/sweep_uniform.sh          # sigma in {0, 0.2, 0.4, 0.6, 0.8, 1.2}
bash a3/sweep_uniform_fine.sh     # finer mapping of the transition region

# Redundancy misfire under correlation, at scale
bash a3/sweep_misfire.sh

# Or a single configuration:
python -m a3.run_synth --noise-mode uniform --sigma 0.2 \
    --d 768 --n-features 16384 --n-outputs 24576 --n-steps 2000 --batch-size 4096 --seeds 0 1
```

Results append to `a3/ledger_a3.jsonl` (gitignored).

## Tests

```bash
pip install -r requirements.txt
pytest -q                  # core suite, pure NumPy (CUDA-dependent tests auto-skip)
```

With a CUDA build of torch installed, the GPU tests in `tests/test_torchcoder.py` and
`tests/test_synthgen.py` are collected and run as well; `conftest.py` skips them
otherwise so the core suite needs no GPU.

## Background

The coder follows the negative-feedback network of Fyfe and the multiple-cause /
sparse-coding line of Foldiak and of Hinton and Ghahramani, with the positivity-plus-noise
construction developed in:

- D. Charles and C. Fyfe, "Modelling multiple-cause structure using rectification
  constraints," *Network: Computation in Neural Systems*, 9(2):167-182, 1998.
- D. Charles, C. Fyfe, D. McDonald, and J. Koetsier, "Unsupervised neural networks for
  the identification of minimum overcomplete basis in visual data," *Neurocomputing*,
  47:119-143, 2002.

A paper describing the experiments in this repository and their relationship to current
sparse-autoencoder practice is in preparation.

## License

MIT. See `LICENSE`.
