# Ours-Plateau balanced rerun

This repository has the Plateau-based proposal methods integrated in Python code, so Kaggle bash cells do not need to patch any Python files.

Main paper-facing method:

- `ours_plateau`: delayed HC micro-modulation on top of `ReduceLROnPlateau`, with the trend-confidence gate disabled.

Ablation methods:

- `plateau`
- `random_plateau`
- `ours_with_gate_plateau`
- `ours_no_hc_plateau`
- `ours_no_noise_norm_plateau`
- `ours_no_clip_plateau`
- `ours_plateau`

Kaggle command:

```bash
bash scripts/run_balanced_ours_plateau.sh
```

Outputs are saved to `/kaggle/working/results_lr_modulator` and zipped manually by the Kaggle cell.
