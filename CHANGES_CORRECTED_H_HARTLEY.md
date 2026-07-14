# Corrected causal EMA–h-Hartley implementation

## Changed files

- `lr_modulator/schedulers.py`
  - restores EMA smoothing before the control map;
  - replaces the delayed four-shift HC-cosine filter with the exact causal signal
    `(h/2) sum_m w_m (u_{t-m}-u_t)`;
  - implements normalized one-sided exponential weights;
  - adds exact even/odd h-Hartley decomposition verification;
  - removes artificial delay while retaining old public statistic keys;
  - fixes the `ours_no_ema_cosine` ablation;
  - updates HC-GAC to reuse the corrected EMA–Hartley signal.
- `lr_modulator/config.py`
  - updates proposal comments and defaults (`m_win=3`);
  - marks `hc_delay` as a deprecated compatibility field.
- `tests/test_hc_convolution.py`
  - adds causality, sign, EMA, clipping, norm-bound, and exact decomposition tests.
- `tests/test_hc_gac_scheduler.py`, `tests/test_scheduler_smoke.py`
  - update causality checks for the zero-delay method.
- `synthetic_compare_old_new.py`
  - evaluates the corrected zero-delay EMA–Hartley signal.
- `README.md` and implementation notes
  - replace obsolete delayed HC-cosine documentation.

## Validation performed

- Python compilation: passed.
- Complete test suite: **52 passed**.
- Training smoke test: **8 lightweight classification/regression runs passed**.
- Synthetic feedback check: passed and wrote `synthetic_compare_output.json`.
- CLI parser/import check: passed.

## Result compatibility

Old experimental results produced by the delayed HC-cosine implementation are not results of the corrected causal EMA–h-Hartley method. Proposed-method and affected ablation experiments must be rerun.
