#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q lr_modulator tests smoke_test.py synthetic_compare_old_new.py run_kaggle.py plot_results.py
python -m pytest tests -q
python smoke_test.py
python synthetic_compare_old_new.py
