# HC-GAC online scheduler patch

## Why this patch was added

The original `ours_*` controller is mathematically clean but often too weak in practice:

- the default modulation cap is small (`gamma=0.05`);
- the fixed gain is very small (`beta_fixed=0.01`);
- the delayed Hartley--cosine convolution needs warm-up steps before becoming active;
- the trend-confidence gate can make `active_mod_rate` close to zero;
- using `plateau` as the base scheduler means the base LR policy can dominate the micro-modulation.

The new `hc_gac_*` methods keep the delayed HC convolution, but add gradient-alignment confirmation. The loss history estimates direction; the gradient cosine alignment decides whether the online LR change is trustworthy.

## New methods

```text
hc_gac_cosine
hc_gac_onecycle
hc_gac_warmup_cosine
hc_gac_plateau
```

## Recommended first experiment

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
  --methods cosine warmup_cosine plateau random_cosine ours_cosine ours_plateau ema_gac_cosine hc_gac_cosine hc_gac_warmup_cosine hc_gac_plateau \
  --m-win 1 \
  --rho 0.8 \
  --ema-gac-dead-zone 0.05 \
  --ema-gac-gamma-up 0.02 \
  --ema-gac-gamma-down 0.06 \
  --ema-gac-confirmation-mode soft
```

## What to report

For each run, inspect the summary statistics:

```text
active_mod_rate
confirmation_rate
positive_delta_rate
negative_delta_rate
delta_mean_abs_final
alignment_mean_abs
trend_score_mean_abs
last_hc_z
hc_warmup_steps
```

If `active_mod_rate < 0.10`, the controller is still too conservative. Try:

```bash
--ema-gac-dead-zone 0.03 --ema-gac-confirmation-mode soft
```

If `delta_mean_abs_final < 0.005`, the modulation is too small. Try:

```bash
--ema-gac-gamma-up 0.03 --ema-gac-gamma-down 0.08
```

If the method is unstable, reduce the down/up bounds:

```bash
--ema-gac-gamma-up 0.01 --ema-gac-gamma-down 0.03
```
