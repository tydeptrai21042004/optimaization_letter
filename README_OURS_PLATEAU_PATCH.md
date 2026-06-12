# Ours-Plateau patch

This patch changes the paper-facing default proposal to `ours_plateau`, which is an alias for:

```text
Ours-no-gate + Plateau = delayed HC micro-modulation on top of ReduceLROnPlateau with trend gate disabled.
```

## New method names

- `ours_plateau`: recommended manuscript/default proposal name.
- `ours_no_gate_plateau`: explicit implementation name; same behavior as `ours_plateau`.
- `random_plateau`: random bounded perturbation on top of Plateau, for reviewer control.

## Plateau-friendly defaults

The global defaults are changed to conservative micro-modulation:

```text
m_win = 1
gamma = 0.05
use_auto_beta = False
beta_fixed = 0.01
target_mean_abs_delta = 0.01
beta_cap = 1.0
random_delta_gamma = 0.05
mod_warmup_steps = 0
```

The causality-safe HC warmup is still enforced internally. With `m_win=1`, the effective HC warmup is 5 steps.

## Example run

```bash
python run_kaggle.py \
  --mode suite \
  --task finetune \
  --dataset oxfordiiitpet \
  --model efficientnet_b0 \
  --epochs 8 \
  --batch-size 32 \
  --lr 0.001 \
  --seeds 0 1 2 \
  --methods plateau random_plateau ours_plateau ours_no_gate_plateau \
  --mod-warmup-steps 0 \
  --m-win 1 \
  --gamma 0.05 \
  --no-auto-beta \
  --beta-fixed 0.01 \
  --no-eval-test-each-epoch
```

## Validation

Validated with:

```text
python -m pytest -q
39 passed, 1 warning
```
