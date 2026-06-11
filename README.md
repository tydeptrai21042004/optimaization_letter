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
| `ours_cosine` | Proposed HC-convolutional modulator on top of cosine |
| `ours_onecycle` | Proposed HC-convolutional modulator on top of one-cycle |
| `ours_warmup_cosine` | Proposed HC-convolutional modulator on top of warmup-cosine |
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
  --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle l4_sgd hyper_sgd dadapt_sgd prodigy ours_cosine ours_onecycle ours_warmup_cosine
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
