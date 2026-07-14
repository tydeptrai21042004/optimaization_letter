# HC-GAC implementation notes

The `hc_gac_*` methods use the corrected causal EMA–\(h\)-Hartley feedback signal and add consecutive-gradient cosine-alignment confirmation.

The loss-side signal is

\[
\bar L_t=\alpha\bar L_{t-1}+(1-\alpha)L_t,
\quad
u_t=\phi(\bar L_t),
\quad
s_t=\frac h2\sum_{m=1}^{M}w_m(u_{t-m}-u_t).
\]

There is no delayed two-sided HC-cosine evaluation. At time `t`, only indices `t-M,...,t` are used. Gradient information observed after backward does not change the learning rate already used by the current optimizer step; the confirmed signal modulates only the next base rate.

Methods:

```text
hc_gac_cosine
hc_gac_onecycle
hc_gac_warmup_cosine
hc_gac_plateau
```

Recommended first check:

```bash
python -m pytest tests/test_hc_gac_scheduler.py -q
```

Useful summary fields:

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
causal_kernel_norm
```
