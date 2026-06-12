#!/usr/bin/env bash
set -euo pipefail

COMMON_CTRL="--mod-warmup-steps 0 --m-win 1 --gamma 0.05 --no-auto-beta --beta-fixed 0.01 --random-delta-gamma 0.05 --no-eval-test-each-epoch"

run_cmd() {
  echo "============================================================"
  echo "$*"
  echo "============================================================"
  eval "$@"
}

# Balanced scratch classification: same method set across datasets.
run_cmd "python run_kaggle.py --mode suite --task scratch --dataset cifar10 --model small_resnet --epochs 20 --batch-size 128 --lr 0.1 --seeds 0 1 --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle random_warmup_cosine random_plateau l4_sgd hyper_sgd ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task scratch --dataset fashionmnist --model small_cnn --epochs 10 --batch-size 128 --lr 0.05 --seeds 0 1 --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle random_warmup_cosine random_plateau l4_sgd hyper_sgd ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task scratch --dataset svhn --model small_cnn --epochs 12 --batch-size 128 --lr 0.05 --seeds 0 1 --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle random_warmup_cosine random_plateau l4_sgd hyper_sgd ours_plateau $COMMON_CTRL"

# Balanced fine-tuning: same method set across datasets.
run_cmd "python run_kaggle.py --mode suite --task finetune --dataset dtd --model mobilenet_v3_small --epochs 5 --batch-size 32 --lr 0.001 --seeds 0 1 --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle random_warmup_cosine random_plateau ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task finetune --dataset flowers102 --model mobilenet_v3_small --epochs 5 --batch-size 32 --lr 0.001 --seeds 0 1 --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle random_warmup_cosine random_plateau ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task finetune --dataset oxfordiiitpet --model efficientnet_b0 --epochs 5 --batch-size 32 --lr 0.001 --seeds 0 1 --methods cosine onecycle warmup_cosine plateau random_cosine random_onecycle random_warmup_cosine random_plateau ours_plateau $COMMON_CTRL"

# Balanced regression.
run_cmd "python run_kaggle.py --mode suite --task regression --dataset synthetic_regression --model small_resnet --epochs 12 --batch-size 32 --lr 0.01 --seeds 0 1 2 --methods cosine warmup_cosine plateau random_cosine random_warmup_cosine random_plateau l4_sgd hyper_sgd ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task regression --dataset synthetic_regression --model tiny_cnn --epochs 12 --batch-size 32 --lr 0.01 --seeds 0 1 2 --methods cosine warmup_cosine plateau random_cosine random_warmup_cosine random_plateau l4_sgd hyper_sgd ours_plateau $COMMON_CTRL"

# Balanced segmentation.
run_cmd "python run_kaggle.py --mode suite --task segmentation --dataset pet_segmentation --model fcn_lite --epochs 4 --batch-size 8 --lr 0.01 --seeds 0 1 --methods cosine warmup_cosine plateau random_cosine random_warmup_cosine random_plateau ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task segmentation --dataset synthetic_segmentation --model fcn_lite --epochs 8 --batch-size 8 --lr 0.01 --seeds 0 1 2 --methods cosine warmup_cosine plateau random_cosine random_warmup_cosine random_plateau ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task segmentation --dataset synthetic_segmentation --model unet_small --epochs 8 --batch-size 8 --lr 0.01 --seeds 0 1 2 --methods cosine warmup_cosine plateau random_cosine random_warmup_cosine random_plateau ours_plateau $COMMON_CTRL"

# New Ours-Plateau ablations.
run_cmd "python run_kaggle.py --mode suite --task finetune --dataset oxfordiiitpet --model efficientnet_b0 --epochs 5 --batch-size 32 --lr 0.001 --seeds 0 1 --methods plateau random_plateau ours_with_gate_plateau ours_no_hc_plateau ours_no_noise_norm_plateau ours_no_clip_plateau ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task scratch --dataset cifar10 --model small_resnet --epochs 15 --batch-size 128 --lr 0.1 --seeds 0 1 --methods plateau random_plateau ours_with_gate_plateau ours_no_hc_plateau ours_no_noise_norm_plateau ours_no_clip_plateau ours_plateau $COMMON_CTRL"
run_cmd "python run_kaggle.py --mode suite --task segmentation --dataset synthetic_segmentation --model unet_small --epochs 6 --batch-size 8 --lr 0.01 --seeds 0 1 --methods plateau random_plateau ours_with_gate_plateau ours_no_hc_plateau ours_no_noise_norm_plateau ours_no_clip_plateau ours_plateau $COMMON_CTRL"

python plot_results.py --results-dir /kaggle/working/results_lr_modulator --max-runs 500
python scripts/export_balanced_tables.py --results-dir /kaggle/working/results_lr_modulator
