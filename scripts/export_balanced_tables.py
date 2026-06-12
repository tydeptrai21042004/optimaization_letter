from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="/kaggle/working/results_lr_modulator")
    args = parser.parse_args()

    res = Path(args.results_dir)
    rows = []
    for p in sorted(res.glob("*_summary.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
            row["_summary_file"] = p.name
            rows.append(row)
        except Exception as exc:
            print(f"[WARN] Could not read {p}: {exc}")

    df = pd.DataFrame(rows)
    out_all = res / "all_run_summaries_BALANCED.csv"
    df.to_csv(out_all, index=False)
    print("Saved:", out_all)

    if df.empty:
        print("[WARN] No summaries found.")
        return

    keys = [
        c
        for c in ["task", "task_type", "dataset", "model", "epochs", "method", "score_name"]
        if c in df.columns
    ]
    metric_cols = [
        c
        for c in [
            "best_val_score",
            "test_score",
            "best_val_acc",
            "test_acc",
            "clip_rate",
            "delta_mean_abs_final",
            "active_mod_rate",
            "beta_eff_mean",
            "time_sec",
        ]
        if c in df.columns
    ]

    agg = df.groupby(keys, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join([a, b]) for a, b in agg.columns]
    agg = agg.reset_index()

    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        count_col = f"{m}_count"
        ci_col = f"{m}_ci95"
        if mean_col in agg.columns and std_col in agg.columns and count_col in agg.columns:
            agg[ci_col] = agg.apply(
                lambda r: 1.96 * r[std_col] / math.sqrt(r[count_col])
                if r[count_col] and r[count_col] > 1
                else float("nan"),
                axis=1,
            )

    out_agg = res / "aggregate_BALANCED.csv"
    agg.to_csv(out_agg, index=False)
    print("Saved:", out_agg)

    def save_subset(name: str, mask) -> None:
        sub = agg[mask].copy()
        out = res / name
        sub.to_csv(out, index=False)
        print("Saved:", out, "rows=", len(sub))

    save_subset("table_scratch_balanced.csv", agg["task"].eq("scratch") if "task" in agg.columns else pd.Series(False, index=agg.index))
    save_subset("table_finetune_balanced.csv", agg["task"].eq("finetune") if "task" in agg.columns else pd.Series(False, index=agg.index))
    save_subset("table_regression_balanced.csv", agg["task"].eq("regression") if "task" in agg.columns else pd.Series(False, index=agg.index))
    save_subset("table_segmentation_balanced.csv", agg["task"].eq("segmentation") if "task" in agg.columns else pd.Series(False, index=agg.index))

    ablation_methods = {
        "plateau",
        "random_plateau",
        "ours_with_gate_plateau",
        "ours_no_hc_plateau",
        "ours_no_noise_norm_plateau",
        "ours_no_clip_plateau",
        "ours_plateau",
    }
    save_subset(
        "table_ours_plateau_ablation_balanced.csv",
        agg["method"].isin(ablation_methods) if "method" in agg.columns else pd.Series(False, index=agg.index),
    )


if __name__ == "__main__":
    main()
