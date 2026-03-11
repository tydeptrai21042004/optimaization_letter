# EMA-Loss LR Modulator Project

This is a modularized version of our Kaggle notebook.

## What is included

- Multiple files instead of one giant notebook cell
- Support for multiple datasets and multiple torchvision models
- Baselines: `constant`, `step`, `cosine`, `onecycle`
- Proposed methods: `ours_cosine`, `ours_onecycle`
- Resume-safe JSON summaries
- Per-run epoch history CSV files
- Aggregated CSV export
- Smoke test that runs without torchvision datasets or internet

## Project structure

```text
lr_modulator_project/
├── README.md
├── requirements.txt
├── run_kaggle.py
├── smoke_test.py
├── lr_modulator/
│   ├── __init__.py
│   ├── config.py
│   ├── runtime.py
│   ├── data.py
│   ├── model_zoo.py
│   ├── schedulers.py
│   ├── engine.py
│   ├── io_utils.py
│   └── experiments.py
└── tests/
    └── test_scheduler_smoke.py
```

## Main split

### `lr_modulator/config.py`
Central experiment configuration.

### `lr_modulator/data.py`
Dataset metadata, transforms, split logic, and dataloaders.

### `lr_modulator/model_zoo.py`
Model builders for:
- `resnet18`
- `resnet34`
- `resnet50`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `vit_b_16`

### `lr_modulator/schedulers.py`
Contains:
- `BatchBaseSchedule`
- `EMALossModulator`
- `Controller`

### `lr_modulator/engine.py`
Training and evaluation loop.

### `lr_modulator/experiments.py`
One-run execution, resume logic, safe-run wrapper, and full grid runner.

### `run_kaggle.py`
Entry point for the full Kaggle experiment.

### `smoke_test.py`
Fast CPU smoke test using synthetic data and a tiny CNN.
This does **not** depend on torchvision datasets, downloads, or Kaggle internet.


## How to run

### 1. Install deps

```bash
pip install -r requirements.txt
```

### 2. Run the smoke test

```bash
python smoke_test.py
```

Expected result:
- prints `Smoke test passed.`
- writes `smoke_test_output.json`

### 3. Run the full Kaggle experiment

```bash
python run_kaggle.py
```

Outputs are written under:
- `./results_lr_modulator` locally, or
- `/kaggle/working/results_lr_modulator` on Kaggle

## Output files

For each run label we get:

- `*_summary.json` — final metrics
- `*_history.csv` — per-epoch metrics

we also get:

- `all_run_summaries.csv`

## Notes

- `torchvision` is imported lazily inside dataset/model functions.
- This means the smoke test can still run even if torchvision is broken locally.
- If pretrained weights are unavailable, the project will skip failing runs when `skip_on_fail=True`.

