# Ours-Plateau configuration

`ours_plateau` remains an alias for the no-gate proposal on top of `ReduceLROnPlateau`.

The corrected proposal is now the causal EMA–\(h\)-Hartley signal

\[
s_t=\frac h2\sum_{m=1}^{M}w_m(u_{t-m}-u_t),
\qquad
\eta_{t+1}=r_{t+1}(1+\delta_t).
\]

It uses no artificial delay. The effective minimum history is `M` steps.

Recommended conservative defaults:

```text
alpha = 0.95
m_win = 3
rho = 0.8
gamma = 0.05
use_auto_beta = False
beta_fixed = 0.01
trend_conf_tau = 0.0
```

Example:

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
  --methods plateau random_plateau ours_no_hc_plateau ours_plateau \
  --mod-warmup-steps 0 \
  --m-win 3 \
  --gamma 0.05 \
  --no-auto-beta \
  --beta-fixed 0.01
```
