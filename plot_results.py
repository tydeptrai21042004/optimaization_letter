from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:180]


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] could not read {path}: {exc}")
        return None
    if df.empty:
        return None
    return df


def _plot_lines(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Iterable[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    y_cols = [c for c in y_cols if c in df.columns and not df[c].isna().all()]
    if x_col not in df.columns or not y_cols:
        return

    fig = plt.figure(figsize=(8, 5))
    for col in y_cols:
        plt.plot(df[x_col], df[col], label=col)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_bar(series: pd.Series, title: str, ylabel: str, out_path: Path) -> None:
    if series.empty:
        return
    fig = plt.figure(figsize=(max(7, 0.45 * len(series)), 5))
    series.plot(kind="bar")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_epoch_history(history_csv: Path, fig_dir: Path) -> List[Path]:
    df = _read_csv(history_csv)
    if df is None:
        return []

    stem = history_csv.name.replace("_history.csv", "")
    out_paths: List[Path] = []

    specs = [
        (["train_loss", "val_loss", "test_loss"], "Loss", "loss"),
        (["train_acc", "val_acc", "test_acc"], "Accuracy", "accuracy"),
        (["last_lr", "last_base_lr"], "Learning rate", "learning_rate"),
        (["last_delta"], "Final batch modulation per epoch", "delta_epoch"),
        (["last_raw"], "Raw trend signal per epoch", "raw_epoch"),
        (["generalization_gap"], "Generalization gap", "generalization_gap"),
    ]

    for cols, ylabel, suffix in specs:
        out_path = fig_dir / f"{_safe_name(stem)}__{suffix}.png"
        _plot_lines(
            df=df,
            x_col="epoch",
            y_cols=cols,
            title=f"{stem}: {ylabel}",
            ylabel=ylabel,
            out_path=out_path,
        )
        if out_path.exists():
            out_paths.append(out_path)

    return out_paths


def plot_batch_history(batch_csv: Path, fig_dir: Path) -> List[Path]:
    df = _read_csv(batch_csv)
    if df is None:
        return []

    stem = batch_csv.name.replace("_batch_history.csv", "")
    df = df.copy()
    df["global_batch"] = range(len(df))
    out_paths: List[Path] = []

    specs = [
        (["lr_used", "base_lr", "lr_next"], "Learning rate", "lr_batch"),
        (["delta"], "Modulation delta", "delta_batch"),
        (["raw"], "Raw trend signal", "raw_batch"),
        (["ema_loss", "train_loss"], "Loss signal", "ema_loss_batch"),
        (["u_signal"], "Control signal", "u_signal_batch"),
        (["beta_eff"], "Effective beta", "beta_eff_batch"),
        (["grad_norm_sq"], "Squared gradient norm", "grad_norm_sq_batch"),
    ]

    for cols, ylabel, suffix in specs:
        out_path = fig_dir / f"{_safe_name(stem)}__{suffix}.png"
        _plot_lines(
            df=df,
            x_col="global_batch",
            y_cols=cols,
            title=f"{stem}: {ylabel}",
            ylabel=ylabel,
            out_path=out_path,
        )
        if out_path.exists():
            out_paths.append(out_path)

    # Rolling clipping frequency, useful for reviewer-facing clipping analysis.
    if "clipped" in df.columns:
        rolling = df[["global_batch", "clipped"]].copy()
        rolling["clip_rate_roll100"] = rolling["clipped"].rolling(100, min_periods=1).mean()
        out_path = fig_dir / f"{_safe_name(stem)}__clipping_frequency.png"
        _plot_lines(
            df=rolling,
            x_col="global_batch",
            y_cols=["clip_rate_roll100"],
            title=f"{stem}: rolling clipping frequency",
            ylabel="rolling clipping frequency",
            out_path=out_path,
        )
        if out_path.exists():
            out_paths.append(out_path)

    return out_paths


def plot_summary_tables(results_dir: Path, fig_dir: Path) -> List[Path]:
    out_paths: List[Path] = []
    aggregate_csv = results_dir / "aggregate_by_method.csv"
    if aggregate_csv.exists():
        df = _read_csv(aggregate_csv)
        if df is not None and "method" in df.columns:
            for metric in ["test_acc_mean", "best_val_acc_mean", "clip_rate_mean", "delta_mean_abs_final_mean"]:
                if metric in df.columns:
                    series = df.set_index("method")[metric].dropna()
                    out_path = fig_dir / f"aggregate__{metric}.png"
                    _plot_bar(series, title=f"Aggregate {metric}", ylabel=metric, out_path=out_path)
                    if out_path.exists():
                        out_paths.append(out_path)

    hparam_rows = []
    for summary_json in results_dir.glob("*_summary.json"):
        try:
            row = pd.read_json(summary_json, typ="series").to_dict()
        except Exception:
            continue
        if row.get("hparam_name") is not None:
            hparam_rows.append(row)
    if hparam_rows:
        df = pd.DataFrame(hparam_rows)
        for hname, sub in df.groupby("hparam_name"):
            if "hparam_value" not in sub.columns or "test_acc" not in sub.columns:
                continue
            grouped = sub.groupby("hparam_value")["test_acc"].mean().sort_index()
            out_path = fig_dir / f"hparam__{_safe_name(str(hname))}__test_acc.png"
            fig = plt.figure(figsize=(7, 5))
            plt.plot(grouped.index.astype(str), grouped.values, marker="o")
            plt.xlabel(str(hname))
            plt.ylabel("mean test accuracy")
            plt.title(f"Hyperparameter sweep: {hname}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
            out_paths.append(out_path)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewer-facing plots from LR-modulator CSV logs.")
    parser.add_argument("--results-dir", type=str, default="./results_lr_modulator")
    parser.add_argument("--fig-dir", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None, help="Optionally limit number of run files plotted.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    made: List[Path] = []

    history_files = sorted(results_dir.glob("*_history.csv"))
    # Exclude batch histories from epoch histories.
    history_files = [p for p in history_files if not p.name.endswith("_batch_history.csv")]
    batch_files = sorted(results_dir.glob("*_batch_history.csv"))

    if args.max_runs is not None:
        history_files = history_files[: args.max_runs]
        batch_files = batch_files[: args.max_runs]

    for path in history_files:
        made.extend(plot_epoch_history(path, fig_dir))
    for path in batch_files:
        made.extend(plot_batch_history(path, fig_dir))
    made.extend(plot_summary_tables(results_dir, fig_dir))

    manifest = fig_dir / "figure_manifest.txt"
    manifest.write_text("\n".join(str(p) for p in made), encoding="utf-8")
    print(f"Generated {len(made)} figure(s).")
    print(f"Figure directory: {fig_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
