# EMA-Loss LR Modulator Benchmark

This repository benchmarks a bounded causal EMA-loss learning-rate modulator against standard schedules and stronger reviewer-requested rivals.

## What is new in this version

The code now includes:

- **Random bounded modulation**: `random_cosine`, `random_onecycle`, `random_warmup_cosine`
- **Loss-based rival**: `l4_sgd`
- **Hypergradient rival**: `hyper_sgd`
- **Modern automatic-LR rivals**: `dadapt_sgd`, `prodigy`
- **Proposed improved method**: bias-corrected EMA, relative trend, optional dead zone, optional variance normalization
- **Ablations**: `ours_no_ema_cosine`, `ours_no_kernel_cosine`, `ours_no_clip_cosine`, `ours_deadzone_cosine`
- **Batch-level logging**: LR used, next LR, raw trend, and modulation `delta`
- **Aggregate statistics**: mean, standard deviation, 95% CI, and paired tests when multiple seeds are available

> For publication-grade D-Adaptation and Prodigy comparisons, install the official optional packages if available: `dadaptation` and `prodigyopt`. If they are missing, the code uses stable internal fallbacks and records `optimizer_impl="internal_fallback"` in the summary JSON/CSV.

## Install

```bash
python -m pip install -r requirements.txt
```

Optional official optimizers:

```bash
python -m pip install dadaptation prodigyopt
```

## Quick smoke test

```bash
python smoke_test.py
```

or with pytest:

```bash
python -m pytest tests
```

## Run one suite

Example: CIFAR-10 / ResNet-18 from scratch with the key rivals:

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 40 \
  --batch-size 128 \
  --lr 0.1 \
  --seeds 0 1 2 3 4 \
  --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle l4_sgd hyper_sgd dadapt_sgd prodigy ours_cosine ours_onecycle ours_warmup_cosine
```

## Run ablations

```bash
python run_kaggle.py \
  --mode ablation \
  --task scratch \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 40 \
  --batch-size 128 \
  --lr 0.1 \
  --seeds 0 1 2 3 4
```

Default ablation methods:

```text
cosine
random_cosine
ours_no_ema_cosine
ours_no_kernel_cosine
ours_no_clip_cosine
ours_deadzone_cosine
ours_cosine
```

## CLI controller overrides

```bash
python run_kaggle.py --mode suite --task scratch --dataset cifar10 --model resnet18 \
  --methods ours_cosine random_cosine \
  --alpha 0.95 --gamma 0.10 --m-win 5 --rho 0.8 --dead-zone-tau 0.005
```

Useful flags:

- `--no-auto-beta`: use fixed beta instead of automatic beta calibration
- `--beta-fixed 0.1`: set fixed beta and disable auto beta
- `--variance-normalize`: normalize raw trend by EMA loss variance
- `--absolute-trend`: use absolute difference instead of relative trend

## Output files

Each run writes:

- `*_summary.json`: final metrics and method settings
- `*_history.csv`: epoch-level train/validation history
- `*_batch_history.csv`: batch-level LR/trend logs

The script also writes:

- `all_run_summaries.csv`
- `aggregate_by_method.csv` with mean/std/95% CI
- `paired_method_tests.csv` for selected paired comparisons

## Recommended paper comparisons

Minimum rival package:

```text
cosine
onecycle
warmup_cosine
plateau
random_cosine
random_onecycle
l4_sgd
hyper_sgd
ours_cosine
ours_onecycle
ours_warmup_cosine
```

Stronger package:

```text
cosine
onecycle
warmup_cosine
warm_restarts
plateau
random_cosine
random_onecycle
random_warmup_cosine
l4_sgd
hyper_sgd
dadapt_sgd
prodigy
ours_cosine
ours_onecycle
ours_warmup_cosine
```

## Why random bounded modulation matters

The reviewer criticized the old theory because any bounded multiplicative LR perturbation would preserve the same generic SGD descent structure. The random bounded baseline directly tests this criticism:

```math
\eta_t = r_t(1 + \epsilon_t), \quad \epsilon_t \sim \mathrm{Uniform}[-\gamma,\gamma].
```

If `ours_cosine` beats `random_cosine`, the improvement is not just from bounded clipping; it comes from the EMA-loss feedback signal.
