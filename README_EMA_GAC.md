# EMA-GAC implementation

This repository now includes **EMA-GAC: Exponential Moving-Average Loss Modulation with Gradient-Alignment Confirmation**.

The controller combines:

1. a fast EMA and a slow EMA of batch loss;
2. a volatility-normalized EMA trend;
3. cosine alignment between consecutive stochastic gradients;
4. strict or soft confirmation of the loss trend by gradient geometry;
5. asymmetric bounded modulation;
6. an optional phase envelope;
7. a strong base schedule such as warmup-cosine.

The LR used by batch `t` is never modified using the loss or gradient from that same batch. The signal observed after batch `t` is applied only to the next base LR:

```text
eta_(t+1) = r_(t+1) * (1 + delta_t)
```

## Added methods

```text
ema_gac_cosine
ema_gac_onecycle
ema_gac_warmup_cosine
ema_gac_plateau
```

For training from scratch, use `ema_gac_warmup_cosine` as the main proposal. For short fine-tuning, compare `ema_gac_plateau` and `ema_gac_cosine`.

## Default EMA-GAC parameters

```yaml
ema_gac_alpha_fast: 0.90
ema_gac_alpha_slow: 0.99
ema_gac_volatility_alpha: 0.95
ema_gac_alignment_alpha: 0.90
ema_gac_beta_up: 1.0
ema_gac_beta_down: 1.5
ema_gac_gamma_up: 0.015
ema_gac_gamma_down: 0.05
ema_gac_dead_zone: 0.10
ema_gac_phase_start: 0.05
ema_gac_phase_end: 0.90
ema_gac_confirmation_mode: strict
```

`gamma_up < gamma_down` makes upward LR changes conservative while allowing stronger reductions when worsening loss and gradient misalignment agree.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Expected test status in this corrected package:

```text
45 passed
```

## Focused CIFAR-10 comparison

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model small_resnet \
  --epochs 30 \
  --seeds 42 1 2 \
  --methods \
    warmup_cosine \
    cosine \
    onecycle \
    random_warmup_cosine \
    ours_warmup_cosine \
    ema_gac_warmup_cosine \
    ema_gac_cosine \
  --sched-warmup-steps 500 \
  --no-eval-test-each-epoch
```

## Stronger EMA-GAC trial

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model small_resnet \
  --epochs 30 \
  --seeds 42 1 2 \
  --methods warmup_cosine ema_gac_warmup_cosine \
  --ema-gac-alpha-fast 0.90 \
  --ema-gac-alpha-slow 0.99 \
  --ema-gac-volatility-alpha 0.95 \
  --ema-gac-alignment-alpha 0.90 \
  --ema-gac-beta-up 1.5 \
  --ema-gac-beta-down 2.0 \
  --ema-gac-gamma-up 0.02 \
  --ema-gac-gamma-down 0.06 \
  --ema-gac-dead-zone 0.05 \
  --ema-gac-phase-start 0.03 \
  --ema-gac-phase-end 0.92 \
  --ema-gac-confirmation-mode strict \
  --no-eval-test-each-epoch
```

## Short-training settings

For 5-10 epoch fine-tuning, the default slow EMA may react too slowly. Start with:

```bash
--ema-gac-alpha-fast 0.80 \
--ema-gac-alpha-slow 0.95 \
--ema-gac-phase-start 0.02 \
--ema-gac-phase-end 0.95
```

## Confirmation modes

```text
strict
```

Allows a positive action only when loss improvement and positive alignment agree, and a negative action only when loss deterioration and negative alignment agree.

```text
soft
```

Uses positive alignment as confidence for the loss-trend direction. This can be more active but is less conservative.

```text
loss_only
```

Ablation without gradient confirmation.

```text
alignment_only
```

Ablation without EMA loss trend.

## Logged diagnostics

Batch CSV files now include:

```text
gradient_alignment
gradient_alignment_ema
trend_score
confirmed_signal
phase_envelope
delta
base_lr
lr_next
```

Run summaries include:

```text
confirmation_rate
positive_delta_rate
negative_delta_rate
delta_mean_abs_final
alignment_mean_abs
trend_score_mean_abs
```

Useful target ranges during development:

```text
delta_mean_abs_final: approximately 0.005 to 0.020
confirmation_rate: approximately 0.10 to 0.60
positive_delta_rate: lower than negative_delta_rate is acceptable
clip_rate: should normally remain low with the tanh controller
```

If `delta_mean_abs_final` is below `0.001`, reduce the dead zone or increase beta/gamma. If it is above `0.03` and validation becomes unstable, reduce `beta_down`, `gamma_down`, or both.

## Gradient-memory control

By default, EMA-GAC stores one previous copy of every parameter gradient. For large models, reduce overhead with:

```bash
--ema-gac-gradient-sample-stride 4 \
--ema-gac-max-gradient-tensors 32
```

This estimates alignment from a reproducible subset of gradient tensors.

## Recommended final experimental protocol

Use validation performance for model and hyperparameter selection. The corrected default no longer evaluates the test set after every epoch. Use at least five seeds for final tables and compare against an amplitude-matched random modulation baseline before claiming that the feedback information itself is useful.
