#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q lr_modulator tests smoke_test.py synthetic_compare_old_new.py run_kaggle.py plot_results.py
python -m pytest tests -q
python - <<'PY'
from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import EMALossModulator
print('Import check passed:', ExperimentConfig.__name__, EMALossModulator.__name__)
PY

echo "All compile, import, and unit checks passed."
