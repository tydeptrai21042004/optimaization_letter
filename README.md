# Hartley--Cosine Convolutional LR Modulator Benchmark

This repository benchmarks a **delayed, noise-normalized, weighted `h`-Hartley--cosine convolutional learning-rate modulator** against standard schedules and reviewer-requested rivals.

The old proposal was an EMA-loss feedback controller. In this corrected version, every `ours_*` method uses the new proposal:

```math
L_t \rightarrow u_t=\varphi(L_t)
\rightarrow z_t=(\kappa *_\gamma u)((t-D)h),\quad D=M+1
\rightarrow q_t=z_{t-1}-z_t
\rightarrow s_t=\frac{q_t}{\sqrt{v_t}+\varepsilon}
\rightarrow \delta_t
\rightarrow \eta_{t+1}=r_{t+1}(1+\delta_t).
```

The delay is essential. The weighted `h`-Hartley--cosine convolution is two-sided. If it is evaluated directly at step `t`, it can require future losses up to `t+M+1`. The implemented version evaluates the convolution at `t-D`, where `D=M+1`, so the largest loss index used is at most `t`. Therefore, the modulation computed after observing `L_t` is safely used only for the next update.

---

## Main methods

| Method name | Meaning |
|---|---|
| `ours_cosine` | Legacy HC-convolutional modulator on top of cosine |
| `ours_onecycle` | Proposed HC-convolutional modulator on top of one-cycle |
| `ours_warmup_cosine` | Proposed HC-convolutional modulator on top of warmup-cosine |
| `hc_gac_cosine`, `hc_gac_onecycle`, `hc_gac_warmup_cosine`, `hc_gac_plateau` | Stronger online HC-convolutional scheduler with gradient-alignment confirmation; recommended when the pure loss-feedback modulation is too weak |
| `ours_no_hc_cosine` | Ablation without Hartley--cosine convolution |
| `ours_no_noise_norm_cosine` | Ablation without noise normalization |
| `ours_no_gate_cosine` | Ablation without trend-confidence gate |
| `ours_no_phi_cosine` | Ablation without bounded loss map `phi` |
| `ours_no_clip_cosine` | Ablation without theoretical clipping, with emergency LR safety only |
| `random_cosine` | Random bounded modulation baseline |
| `l4_sgd` | L4-style loss-based LR rival |
| `hyper_sgd` | Hypergradient LR rival |
| `dadapt_sgd`, `prodigy` | Automatic LR rivals, with internal fallback if optional packages are absent |

Backward-compatible old names such as `ours_no_kernel_cosine`, `ours_no_ema_cosine`, and `ours_deadzone_cosine` still run, but the names above are recommended for the revised manuscript.

---

## Install

```bash
python -m pip install -r requirements.txt
```

Optional official optimizer packages:

```bash
python -m pip install dadaptation prodigyopt
```

If these optional packages are not installed, the code uses stable internal fallback implementations and records this in the summary files.

---

## One-command validation

Run this after extracting the repository:

```bash
bash run_all_checks.sh
```

This command performs:

```bash
python -m compileall -q lr_modulator tests smoke_test.py synthetic_compare_old_new.py run_kaggle.py plot_results.py
python -m pytest tests -q
python smoke_test.py
python synthetic_compare_old_new.py
```

For a faster unit-test-only check:

```bash
python -m pytest tests -q
```

For only the new Hartley--cosine convolution tests:

```bash
python -m pytest tests/test_hc_convolution.py -q
```

---

## Added test cases

The repository now includes additional HC-specific tests in:

```text
tests/test_hc_convolution.py
```

These tests check:

1. **Delayed adaptedness**: the delayed convolution never uses a loss index greater than the current step.
2. **Direct noncausality proof by index**: direct evaluation would use future losses, while delayed evaluation does not.
3. **Kernel normalization**: a constant input remains constant after the HC convolution, after warm-up.
4. **Incomplete-history protection**: the convolution returns `None` when the required history is unavailable.
5. **Clipping safety**: `|delta_t| <= gamma` even when the raw score is intentionally made very large.
6. **Trend-confidence gate**: flat loss produces no active modulation.
7. **No-HC ablation**: the `ours_no_hc_cosine` path uses a simple causal trend and does not need future history.

Existing tests in `tests/test_scheduler_smoke.py` still check controller execution for the proposed methods, random modulation, L4, HyperSGD, D-Adaptation fallback, Prodigy fallback, old aliases, and the shifted next-step LR convention.

Additional baseline-correctness tests are now included in:

```text
tests/test_baseline_correctness.py
tests/test_full_runnability_matrix.py
```

These tests check:

1. **Closed-form LR schedules**: `constant`, `step`, `cosine`, and `warmup_cosine` match their expected formulas.
2. **Plateau behavior**: `plateau` changes LR only from epoch-level validation metrics.
3. **Baseline isolation**: standard baselines are independent of the loss sequence and always log `last_delta=0`.
4. **Random baseline bounds**: `random_cosine` always satisfies `|delta| <= random_delta_gamma` and `lr = base_lr * (1 + delta)`.
5. **L4 formula check**: the first L4 update is compared against the closed-form loss-gap / gradient-norm rule.
6. **HyperSGD formula check**: the second hypergradient update is compared against `lr_t + hyper_lr * <g_t, g_{t-1}>`.
7. **Optimizer-only baselines**: `adamw`, `dadapt_sgd`, and `prodigy` instantiate correctly and record whether an official or internal fallback implementation is used.
8. **Full runnability matrix**: all classification baseline/proposed methods run one training epoch; regression and segmentation method groups also run one epoch.
9. **Task-aware summaries**: `fit()` returns `task_type`, `score_name`, `best_val_score`, and `test_score` for classification, regression, and segmentation.

---

## Quick smoke test

```bash
python smoke_test.py
```

The smoke test runs lightweight synthetic training loops and writes:

```text
smoke_test_output.json
```

---

## Synthetic old-vs-new comparison

This script compares the old EMA-style feedback against the new delayed HC-convolution feedback on a controlled noisy loss signal:

```bash
python synthetic_compare_old_new.py
```

The output is saved to:

```text
synthetic_compare_output.json
```

This is only a signal-quality sanity check. It is not a replacement for real training experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet, or noisy-label settings.

---

## Run one training suite

Example: CIFAR-10 / ResNet-18 from scratch with the key schedules and rivals:

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
  --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle l4_sgd hyper_sgd dadapt_sgd prodigy ours_cosine ours_onecycle ours_warmup_cosine hc_gac_cosine hc_gac_warmup_cosine hc_gac_plateau
```

For a quick debug run:

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 2 \
  --batch-size 128 \
  --lr 0.1 \
  --seeds 0 \
  --methods cosine random_cosine ours_cosine
```

---

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
ours_no_hc_cosine
ours_no_noise_norm_cosine
ours_no_gate_cosine
ours_no_clip_cosine
ours_cosine
```

Recommended paper ablation table:

| Comparison | Purpose |
|---|---|
| `cosine` vs `ours_cosine` | full method versus base schedule |
| `random_cosine` vs `ours_cosine` | structured feedback versus random bounded perturbation |
| `ours_no_hc_cosine` vs `ours_cosine` | contribution of HC convolution |
| `ours_no_noise_norm_cosine` vs `ours_cosine` | contribution of noise normalization |
| `ours_no_gate_cosine` vs `ours_cosine` | contribution of trend-confidence gating |
| `ours_no_clip_cosine` vs `ours_cosine` | role of theoretical clipping |

---

## CLI controller overrides

Example with custom HC-controller settings:

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model resnet18 \
  --methods ours_cosine random_cosine \
  --gamma 0.10 \
  --m-win 5 \
  --rho 0.8 \
  --trend-conf-tau 0.25 \
  --hc-h 1.0
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--m-win 5` | HC kernel half-window `M` |
| `--rho 0.8` | HC kernel decay |
| `--hc-h 1.0` | time-scale step `h` |
| `--gamma 0.10` | clipping bound for `delta_t` |
| `--trend-conf-tau 0.25` | active modulation threshold |
| `--dead-zone-tau 0.25` | legacy alias for trend-confidence threshold |
| `--variance-normalize` | explicitly enable noise normalization |
| `--no-auto-beta` | use fixed beta instead of automatic calibration |
| `--beta-fixed 0.1` | set fixed beta |
| `--absolute-trend` | use absolute trend instead of relative trend |
| `--no-eval-test-each-epoch` | faster large-scale run, no test evaluation every epoch |

---

## Output files

Each run writes files under `results_lr_modulator/`:

| File | Content |
|---|---|
| `*_summary.json` | final metrics and method settings |
| `*_history.csv` | epoch-level train/validation/test history |
| `*_batch_history.csv` | batch-level LR/trend/modulation logs |
| `all_run_summaries.csv` | all run summaries combined |
| `aggregate_by_method.csv` | mean/std/95% CI per method |
| `paired_method_tests.csv` | selected paired comparisons |

Batch-level logs include:

```text
base_lr, beta_eff, ema_loss, u_signal, clipped, emergency_clipped, grad_norm_sq
```

Note: `ema_loss` is kept as a legacy column name. In the corrected method it stores the delayed HC-filtered value `z_t`.

---

## Plot generation

After a run, create reviewer-facing plots from saved CSV logs:

```bash
python plot_results.py --results-dir ./results_lr_modulator
```

The script generates plots for training/validation/test curves, learning-rate trajectories, modulation `delta_t`, raw HC trend signal, bounded control signal, clipping frequency, gradient norm, and aggregate summaries.

---

## Recommended paper comparisons

Minimum comparison package:

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

Stronger comparison package:

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

For the revised paper, prioritize harder settings where loss-feedback should matter:

```text
CIFAR-10
CIFAR-100
CIFAR-10 with noisy labels
small-batch training
high initial LR stress tests
```

---

## Why random bounded modulation matters

The reviewer/editor criticism was that generic bounded multiplicative LR perturbations can preserve the same SGD descent structure. The random bounded baseline tests this directly:

```math
\eta_t=r_t(1+\epsilon_t),\qquad \epsilon_t\sim\mathrm{Uniform}[-\gamma,\gamma].
```

If `ours_cosine` beats `random_cosine`, the improvement is not only due to clipping. It comes from the structured delayed Hartley--cosine convolutional loss-feedback signal.

---

## Hyperparameter sweep

Run one-factor-at-a-time sensitivity analysis:

```bash
python run_kaggle.py \
  --mode hparam \
  --task scratch \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 40 \
  --seeds 0 1 2 3 4 \
  --methods ours_cosine
```

Default grids:

```text
alpha: 0.80, 0.90, 0.95, 0.99  # retained only for compatibility
M:     3, 5, 10, 20
rho:   0.3, 0.5, 0.7, 0.9
beta:  0.05, 0.10, 0.20, 0.50
gamma: 0.05, 0.10, 0.20, 0.30
```

For beta sweeps, `use_auto_beta` is disabled automatically so the fixed beta value is actually used.

---

## Multi-dataset and multi-task update

This version is no longer limited to image classification. The training engine is now **task-aware** and automatically selects the correct loss and metric from the dataset registry.

### Supported task types

| Task type | Loss | Main validation/test metric | Example datasets |
|---|---|---|---|
| `classification` | cross entropy | accuracy | `cifar10`, `cifar100`, `svhn`, `mnist`, `fashionmnist`, `stl10`, `gtsrb`, `flowers102`, `food101`, `oxfordiiitpet`, `dtd`, `eurosat`, `country211`, `caltech101`, `caltech256`, `synthetic_classification` |
| `regression` | mean squared error | RMSE | `synthetic_regression` |
| `segmentation` | pixelwise cross entropy | pixel accuracy | `synthetic_segmentation`, `voc_segmentation`, `pet_segmentation` |

The old classification fields such as `best_val_acc` and `test_acc` are still saved for backward compatibility. New generic fields are also saved: `task_type`, `score_name`, `best_val_score`, and `test_score`.

### Expanded backbone and architecture support

The model zoo now supports both **offline built-in models** and many **torchvision backbones**. Built-in models are useful for debugging, synthetic experiments, and CPU/Kaggle smoke tests because they do not require pretrained weight downloads. Torchvision models are intended for full real-dataset experiments.

#### Built-in classification/regression models

| Model | Main use |
|---|---|
| `tiny_mlp` | Very fast MLP baseline for synthetic/debug runs |
| `tiny_cnn` | Minimal CNN smoke-test model |
| `small_cnn` | Stronger small CNN baseline |
| `depthwise_cnn` | MobileNet-style depthwise-separable CNN |
| `small_resnet` | CIFAR-style residual network |
| `wide_small_resnet` | Wider/deeper CIFAR-style residual network |
| `mini_vit` | Lightweight ViT-style transformer with convolutional patch embedding |

These models work for both `classification` and `regression`; only the final prediction head changes.

#### Built-in segmentation models

| Model | Main use |
|---|---|
| `tiny_unet` | Very fast synthetic segmentation smoke tests |
| `unet_small` | U-Net baseline with skip connections |
| `fcn_lite` | Lightweight fully convolutional segmentation baseline |
| `deeplab_lite` | Tiny DeepLab-style atrous-context baseline |

#### Torchvision classification backbones

Supported names include ResNet/ResNeXt/Wide-ResNet, DenseNet, EfficientNet-B0--B7, EfficientNet-V2, MobileNet, MNASNet, ShuffleNet, SqueezeNet, ConvNeXt, RegNet, MaxViT, ViT, Swin/Swin-V2, AlexNet, VGG, GoogLeNet, and Inception-V3. Examples:

```text
resnet18, resnet50, resnet101, resnext50_32x4d, wide_resnet50_2,
densenet121, densenet201, mobilenet_v3_small, efficientnet_b0, efficientnet_b7,
efficientnet_v2_s, convnext_tiny, convnext_base, regnet_y_400mf,
shufflenet_v2_x1_0, squeezenet1_1, vit_b_16, swin_t, swin_v2_t,
alexnet, vgg16_bn, googlenet, inception_v3
```

The model builder replaces `fc`, `classifier`, `head`, or `heads` prediction heads automatically, so the same backbone can be used for different class counts or for scalar regression. For CIFAR-style scratch ResNets, the first convolution/max-pool are adapted safely to 32x32 input.

#### Torchvision segmentation architectures

```text
fcn_resnet50, fcn_resnet101,
deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large,
lraspp_mobilenet_v3_large
```

The segmentation builder replaces the pixel classifier head automatically, including LR-ASPP's low/high classifiers.

To list all currently supported names from Python:

```bash
python - <<'PY'
from lr_modulator.model_zoo import list_supported_models
print('classification/regression:', list_supported_models('classification'))
print('segmentation:', list_supported_models('segmentation'))
PY
```


### Backbone smoke-test commands

Built-in classification/regression model smoke run:

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset synthetic_classification \
  --model small_resnet \
  --epochs 1 \
  --batch-size 8 \
  --lr 0.01 \
  --seeds 0 \
  --methods cosine ours_cosine \
  --no-eval-test-each-epoch
```

Built-in segmentation architecture smoke run:

```bash
python run_kaggle.py \
  --mode suite \
  --task segmentation \
  --dataset synthetic_segmentation \
  --model unet_small \
  --epochs 1 \
  --batch-size 4 \
  --lr 0.01 \
  --seeds 0 \
  --methods cosine ours_cosine \
  --no-eval-test-each-epoch
```

Real pretrained backbone example:

```bash
python run_kaggle.py \
  --mode suite \
  --task finetune \
  --dataset oxfordiiitpet \
  --model efficientnet_b0 \
  --epochs 5 \
  --batch-size 32 \
  --lr 0.001 \
  --seeds 0 \
  --methods cosine ours_cosine
```

### Run task-aware examples

Classification, unchanged from the old workflow:

```bash
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 2 \
  --batch-size 128 \
  --lr 0.1 \
  --seeds 0 \
  --methods cosine random_cosine ours_cosine
```

Regression smoke run:

```bash
python run_kaggle.py \
  --mode suite \
  --task regression \
  --dataset synthetic_regression \
  --model tiny_cnn \
  --epochs 1 \
  --batch-size 8 \
  --lr 0.01 \
  --seeds 0 \
  --methods cosine random_cosine ours_cosine \
  --no-eval-test-each-epoch
```

Segmentation smoke run:

```bash
python run_kaggle.py \
  --mode suite \
  --task segmentation \
  --dataset synthetic_segmentation \
  --model tiny_unet \
  --epochs 1 \
  --batch-size 4 \
  --lr 0.01 \
  --seeds 0 \
  --methods cosine random_cosine ours_cosine \
  --no-eval-test-each-epoch
```

Real segmentation example:

```bash
python run_kaggle.py \
  --mode suite \
  --task segmentation \
  --dataset pet_segmentation \
  --model fcn_resnet50 \
  --epochs 5 \
  --batch-size 4 \
  --lr 0.01 \
  --seeds 0 \
  --methods cosine ours_cosine
```

### New tests

Additional tests were added in:

```text
tests/test_task_types.py
tests/test_model_zoo_backbones.py
tests/conftest.py
```

They check that:

1. the dataset registry contains classification, regression, and segmentation datasets;
2. synthetic regression trains with `tiny_cnn` and reports RMSE;
3. synthetic segmentation trains with `tiny_unet` and reports pixel accuracy;
4. all built-in classification/regression backbones produce the correct output shape;
5. all built-in segmentation architectures produce pixel logits with shape `[B, C, H, W]`;
6. torchvision-like `fc`, `classifier`, `head`, `heads`, and convolutional classifier heads are replaced correctly;
7. the supported model registry includes the newly added backbones.

Run everything with:

```bash
bash run_all_checks.sh
```

Current validation result on this patched repo:

```text
24 passed
Smoke test passed.
synthetic_compare_old_new.py completed and printed the summary JSON.
```

## EMA-GAC extension

The corrected repository includes EMA-GAC methods (`ema_gac_warmup_cosine`, `ema_gac_cosine`, `ema_gac_onecycle`, and `ema_gac_plateau`). See [`README_EMA_GAC.md`](README_EMA_GAC.md) for the equations, parameters, tests, and reproducible commands.
