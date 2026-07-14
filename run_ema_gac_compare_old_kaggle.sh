#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Fair Kaggle comparison: new EMA-GAC vs old HC proposal
# No source-code patching and no inline Python code.
# Everything is executed directly from the GitHub repository.
# ============================================================

REPO_URL="https://github.com/tydeptrai21042004/optimaization_letter.git"
WORK_ROOT="/kaggle/working"
REPO_DIR="${WORK_ROOT}/optimaization_letter"
RESULTS_DIR="${WORK_ROOT}/results_lr_modulator"
ZIP_PATH="${WORK_ROOT}/results_ema_gac_vs_old.zip"

cd "${WORK_ROOT}"
rm -rf "${REPO_DIR}" "${RESULTS_DIR}"
rm -f "${ZIP_PATH}"

printf '\n============================================================\n'
printf 'EMA-GAC vs old HC proposal\n'
printf 'Repository: %s\n' "${REPO_URL}"
printf 'No ZIP upload, no source patching, no inline Python patch\n'
printf '============================================================\n\n'

git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

printf '\n============================================================\n'
printf 'Git revision\n'
printf '============================================================\n'
git rev-parse HEAD
git log -1 --oneline

printf '\n============================================================\n'
printf 'Install dependencies\n'
printf '============================================================\n'
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt pytest pandas matplotlib

# These are optional and are not required by EMA-GAC or the old HC method.
python -m pip install -q dadaptation prodigyopt || \
  echo '[WARN] Optional dadaptation/prodigyopt installation failed.'

printf '\n============================================================\n'
printf 'Validate repository before training\n'
printf '============================================================\n'
python -m compileall -q \
  lr_modulator \
  tests \
  run_kaggle.py \
  plot_results.py \
  scripts/export_balanced_tables.py

python -m pytest tests -q

printf '\n============================================================\n'
printf 'Run fair paired comparison\n'
printf 'Same dataset, model, seeds, epochs, optimizer and base LR\n'
printf '============================================================\n'

# Fair comparison groups:
#   warmup_cosine           : unmodified strong base schedule
#   random_warmup_cosine    : random bounded-control baseline
#   ours_warmup_cosine      : corrected causal EMA--h-Hartley proposal
#   ema_gac_warmup_cosine   : new EMA-GAC proposal
#   cosine / onecycle       : additional strong schedule baselines
#
# Five paired seeds are used so every method sees the same initializations
# and data-order seeds. Test data are evaluated only after validation-based
# checkpoint selection.
python run_kaggle.py \
  --mode suite \
  --task scratch \
  --dataset cifar10 \
  --model small_resnet \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.1 \
  --seeds 42 1 2 3 4 \
  --methods \
    cosine \
    onecycle \
    warmup_cosine \
    random_warmup_cosine \
    ours_warmup_cosine \
    ema_gac_warmup_cosine \
  --sched-warmup-steps 500 \
  --random-delta-gamma 0.05 \
  --m-win 1 \
  --gamma 0.05 \
  --no-auto-beta \
  --beta-fixed 0.01 \
  --ema-gac-alpha-fast 0.90 \
  --ema-gac-alpha-slow 0.99 \
  --ema-gac-volatility-alpha 0.95 \
  --ema-gac-alignment-alpha 0.90 \
  --ema-gac-beta-up 1.0 \
  --ema-gac-beta-down 1.5 \
  --ema-gac-gamma-up 0.015 \
  --ema-gac-gamma-down 0.05 \
  --ema-gac-dead-zone 0.10 \
  --ema-gac-phase-start 0.05 \
  --ema-gac-phase-end 0.90 \
  --ema-gac-confirmation-mode strict \
  --ema-gac-gradient-sample-stride 1 \
  --ema-gac-max-gradient-tensors 0 \
  --no-eval-test-each-epoch

printf '\n============================================================\n'
printf 'Generate repository plots and aggregate tables\n'
printf '============================================================\n'
python plot_results.py \
  --results-dir "${RESULTS_DIR}" \
  --max-runs 1000

python scripts/export_balanced_tables.py \
  --results-dir "${RESULTS_DIR}"

AGG_FILE="${RESULTS_DIR}/aggregate_BALANCED.csv"
ALL_FILE="${RESULTS_DIR}/all_run_summaries_BALANCED.csv"

if [[ ! -s "${AGG_FILE}" ]]; then
  echo "ERROR: Aggregate result file was not created: ${AGG_FILE}"
  exit 1
fi

printf '\n============================================================\n'
printf 'FINAL COMPARISON TABLE\n'
printf 'Higher test score is better for CIFAR-10 classification\n'
printf '============================================================\n'

# Print a compact table directly from the repository-generated aggregate CSV.
# This is reporting only; it does not modify source code or results.
awk -F',' '
NR == 1 {
  for (i = 1; i <= NF; i++) {
    if ($i == "method") method_col = i
    if ($i == "best_val_score_mean") val_col = i
    if ($i == "best_val_score_std") val_std_col = i
    if ($i == "test_score_mean") test_col = i
    if ($i == "test_score_std") test_std_col = i
    if ($i == "test_score_count") count_col = i
    if ($i == "delta_mean_abs_final_mean") delta_col = i
    if ($i == "clip_rate_mean") clip_col = i
  }
  printf "%-26s %12s %12s %12s %12s %8s %12s %10s\n", \
         "METHOD", "VAL_MEAN", "VAL_STD", "TEST_MEAN", "TEST_STD", "SEEDS", "MEAN_|DELTA|", "CLIP_RATE"
  printf "%-26s %12s %12s %12s %12s %8s %12s %10s\n", \
         "--------------------------", "------------", "------------", "------------", "------------", "--------", "------------", "----------"
  next
}
{
  method = $(method_col)
  if (method == "cosine" ||
      method == "onecycle" ||
      method == "warmup_cosine" ||
      method == "random_warmup_cosine" ||
      method == "ours_warmup_cosine" ||
      method == "ema_gac_warmup_cosine") {
    val = (val_col ? $(val_col) : "NA")
    val_std = (val_std_col ? $(val_std_col) : "NA")
    test = (test_col ? $(test_col) : "NA")
    test_std = (test_std_col ? $(test_std_col) : "NA")
    seeds = (count_col ? $(count_col) : "NA")
    delta = (delta_col ? $(delta_col) : "NA")
    clip = (clip_col ? $(clip_col) : "NA")
    printf "%-26s %12s %12s %12s %12s %8s %12s %10s\n", \
           method, val, val_std, test, test_std, seeds, delta, clip
  }
}
' "${AGG_FILE}"

printf '\n============================================================\n'
printf 'DIRECT NEW-vs-OLD VERDICT\n'
printf '============================================================\n'

awk -F',' '
NR == 1 {
  for (i = 1; i <= NF; i++) {
    if ($i == "method") method_col = i
    if ($i == "test_score_mean") test_col = i
  }
  next
}
{
  if ($(method_col) == "ours_warmup_cosine") old_score = $(test_col) + 0
  if ($(method_col) == "ema_gac_warmup_cosine") new_score = $(test_col) + 0
}
END {
  if (old_score == "" || new_score == "") {
    print "ERROR: Could not find both old and new proposal rows."
    exit 1
  }

  difference = new_score - old_score
  printf "Old HC proposal test mean : %.6f\n", old_score
  printf "New EMA-GAC test mean     : %.6f\n", new_score
  printf "EMA-GAC minus old HC      : %+.6f\n", difference

  if (difference > 0) {
    print "RESULT: EMA-GAC BEATS the old HC proposal on mean test score."
  } else if (difference < 0) {
    print "RESULT: EMA-GAC DOES NOT beat the old HC proposal in this run."
  } else {
    print "RESULT: EMA-GAC and the old HC proposal are tied on mean test score."
  }
}
' "${AGG_FILE}"

printf '\nImportant: a larger mean alone is not proof of statistical superiority.\n'
printf 'Use the five per-seed rows in:\n  %s\n' "${ALL_FILE}"
printf 'for paired statistical testing before making a manuscript claim.\n'

printf '\n============================================================\n'
printf 'Package outputs\n'
printf '============================================================\n'
cd "${WORK_ROOT}"
zip -qr "${ZIP_PATH}" "$(basename "${RESULTS_DIR}")"

printf '\n============================================================\n'
printf 'DONE\n'
printf 'Results directory : %s\n' "${RESULTS_DIR}"
printf 'Aggregate table   : %s\n' "${AGG_FILE}"
printf 'All seed results  : %s\n' "${ALL_FILE}"
printf 'ZIP archive       : %s\n' "${ZIP_PATH}"
printf '============================================================\n'
