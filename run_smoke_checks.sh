#!/usr/bin/env bash
set -euo pipefail
python smoke_test.py
python synthetic_compare_old_new.py
