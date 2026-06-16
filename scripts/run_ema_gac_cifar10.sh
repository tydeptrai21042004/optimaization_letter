#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pytest -q

python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model small_resnet \
  --epochs 30 \
  --seeds 42 1 2 \
  --methods \
    cosine \
    warmup_cosine \
    onecycle \
    random_warmup_cosine \
    ours_warmup_cosine \
    ema_gac_cosine \
    ema_gac_warmup_cosine \
  --ema-gac-alpha-fast 0.90 \
  --ema-gac-alpha-slow 0.99 \
  --ema-gac-volatility-alpha 0.95 \
  --ema-gac-alignment-alpha 0.90 \
  --ema-gac-beta-up 1.0 \
  --ema-gac-beta-down 1.5 \
  --ema-gac-gamma-up 0.015 \
  --ema-gac-gamma-down 0.05 \
  --ema-gac-dead-zone 0.10 \
  --ema-gac-confirmation-mode strict \
  --no-eval-test-each-epoch
