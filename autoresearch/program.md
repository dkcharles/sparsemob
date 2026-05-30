# autoresearch program — nfnet experiments

You are the research agent. Iterate experiments by editing **only** a knob config
and running `run_trial.py`. You may NOT edit `nfnet/` or change the mechanism.

## Locked mechanism (do not change)
- Negative-feedback network with a positive soft-threshold output non-linearity.
- Experiment `mob`: uniform-Gaussian output noise on. Experiment `signed_bars`:
  signed-bars data, noise off by default.

## Knobs you may vary (TrialConfig)
`tau`, `lam`, `eta0`, `sigma`, `n_outputs`, `weight_init`, `n_steps`, `seeds`.
Search priors for `mob`: `sigma in [0.02, 0.3]` (primary lever),
`tau in [0.5,1.5]`, `lam in [2,6]`, `eta0 in [0.02,0.1]`, `n_steps in [50k,150k]`.

## Current objective: `mob` (Minimum Overcomplete Basis width-selection)
With `n_outputs=24` on 16-cause bars, drive `seed_pass_fraction -> 1.0` at the
lowest `n_steps`. A trial passes a seed when exactly 16 outputs are active and all
16 bars are recovered. Always run one `sigma=0.0` contrast trial per batch.

## Next objective: `signed_bars` (non-negativity fragmentation)
On signed (+/-1) bars, the positive-only code must use a separate output for the
+bar and -bar of each cause. Start at `n_outputs=16` and raise it; the target is
to recover all `2*(2*size)=32` signed features (`seed_pass_fraction -> 1.0`).
Noise is off by default for this experiment (`sigma=0.0`). The prediction is that
~32 outputs are needed; record the minimum `n_outputs` that achieves full recovery.

## Comparison objective: `signed_bars_signed` (sign-preserving code)

Same signed (+/-1) bars, but the sign-preserving `soft_shrink` non-linearity (outputs
may be negative). One atom can code both the +bar and -bar of a cause, so the target is
to recover all `2*size=16` causes up to sign (|cosine|), predicted with only ~16 atoms
(half the non-negative code's 32). Start `n_outputs=8` and raise it. Noise off by default.

## Scaling objective: `synth` (SynthSAEBench-style features)

Continuous, superposed features (random unit-vector dictionary, Zipfian firing) at
d-dim activations. Over-provision the coder (n_outputs > n_features) and ask whether
output noise prunes the active set toward n_features while keeping a high MCC
(Hungarian |cos|) against the ground-truth dictionary. Knobs add `d`, `n_features`,
`batch_size`; training is minibatched. Default scale d=128, n_features=512,
n_outputs=768. This tests whether the noise->MOB result transfers beyond binary bars.

## The loop (batched check-ins)
1. Read the per-experiment baseline (configs/<experiment>_baseline.json) and the last rows of `ledger/log.md`.
2. Propose the next config with a one-paragraph hypothesis (what it tests, why).
3. Run it, e.g.:
   `python -m autoresearch.run_trial --experiment mob --sigma 0.08 --n-steps 80000 --rationale "test lower sigma floor"`
4. The runner appends a ledger row and updates the baseline if KEPT.
5. After ~8-12 trials, summarise findings in `ledger/log.md` and PAUSE for human review.

## Keep/discard rule (applied automatically by run_trial)
Keep if `seed_pass_fraction` improves; tie-break on lower `primary_score`, then
fewer `n_steps`. Otherwise discard (logged with reason).
