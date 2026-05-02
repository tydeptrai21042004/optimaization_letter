from __future__ import annotations

import argparse
import os

from lr_modulator.config import ExperimentConfig
from lr_modulator.experiments import (
    paired_method_tests,
    run_ablation_suite,
    run_all,
    run_method_suite,
    summarize_replicates,
)
from lr_modulator.io_utils import save_aggregate_csv
from lr_modulator.runtime import get_device, validate_methods


def print_summary_table(all_summaries):
    print("\n" + "=" * 190)
    print(
        f"{'dataset':<14}{'model':<20}{'method':<26}{'seed':>6}"
        f"{'task':>12}{'epochs':>8}{'val':>10}{'test':>10}"
        f"{'clip':>10}{'|d|':>10}{'beta_eff':>10}{'impl':>20}{'time(s)':>10}"
    )
    print("-" * 190)

    for s in all_summaries:
        clip = s.get("clip_rate", None)
        dabs = s.get("delta_mean_abs_final", None)
        beta_eff = s.get("beta_eff_mean", None)
        impl = str(s.get("optimizer_impl", "-"))[:19]

        print(
            f"{s['dataset']:<14}"
            f"{s['model']:<20}"
            f"{s['method']:<26}"
            f"{s['seed']:>6}"
            f"{s.get('task', '-'):>12}"
            f"{s.get('epochs', '-'):>8}"
            f"{s['best_val_acc']:>10.4f}"
            f"{s['test_acc']:>10.4f}"
            f"{('-' if clip is None else f'{clip:.4f}'):>10}"
            f"{('-' if dabs is None else f'{dabs:.4f}'):>10}"
            f"{('-' if beta_eff is None else f'{beta_eff:.3f}'):>10}"
            f"{impl:>20}"
            f"{s['time_sec']:>10.1f}"
        )

    print("=" * 190)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        type=str,
        choices=["full", "suite", "ablation"],
        default="full",
        help="full = run paper grid, suite = run selected methods, ablation = run reviewer ablation methods",
    )

    p.add_argument(
        "--task",
        type=str,
        choices=["scratch", "finetune"],
        default=None,
        help="Used in --mode suite/ablation",
    )

    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Example: --seeds 0 1 2 3 4",
    )

    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Example: --methods cosine random_cosine l4_sgd hyper_sgd "
            "dadapt_sgd prodigy ours_cosine ours_onecycle ours_warmup_cosine"
        ),
    )

    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--m-win", type=int, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--beta-fixed", type=float, default=None)
    p.add_argument("--no-auto-beta", action="store_true")
    p.add_argument("--dead-zone-tau", type=float, default=None)
    p.add_argument("--variance-normalize", action="store_true")
    p.add_argument("--absolute-trend", action="store_true")

    return p


def apply_cli_overrides(config: ExperimentConfig, args) -> None:
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.gamma is not None:
        config.gamma = args.gamma
        config.random_delta_gamma = args.gamma
    if args.m_win is not None:
        config.m_win = args.m_win
    if args.rho is not None:
        config.rho = args.rho
    if args.beta_fixed is not None:
        config.beta_fixed = args.beta_fixed
        config.use_auto_beta = False
    if args.no_auto_beta:
        config.use_auto_beta = False
    if args.dead_zone_tau is not None:
        config.dead_zone_tau = args.dead_zone_tau
    if args.variance_normalize:
        config.variance_normalize = True
    if args.absolute_trend:
        config.relative_trend = False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = ExperimentConfig()
    apply_cli_overrides(config, args)
    device, _ = get_device()

    if args.methods is not None:
        args.methods = validate_methods(args.methods)

    if args.mode in {"suite", "ablation"}:
        if args.task is None or args.dataset is None or args.model is None:
            raise ValueError(f"--mode {args.mode} requires --task, --dataset, and --model")

        if args.mode == "suite":
            all_summaries = run_method_suite(
                config=config,
                device=device,
                task=args.task,
                dataset=args.dataset,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_lr=args.lr,
                seeds=args.seeds,
                methods=args.methods,
            )
        else:
            if args.methods is not None:
                config.ablation_methods = args.methods
            all_summaries = run_ablation_suite(
                config=config,
                device=device,
                task=args.task,
                dataset=args.dataset,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_lr=args.lr,
                seeds=args.seeds,
            )
    else:
        if args.methods is not None:
            print("[INFO] --methods is ignored in --mode full; using config-defined grids.")
        all_summaries = run_all(config, device)

    csv_path = os.path.join(config.save_dir, "all_run_summaries.csv")
    save_aggregate_csv(csv_path, all_summaries)

    aggregate_path = os.path.join(config.save_dir, "aggregate_by_method.csv")
    summarize_replicates(all_summaries, aggregate_path)

    paired_path = os.path.join(config.save_dir, "paired_method_tests.csv")
    paired_method_tests(all_summaries, out_path=paired_path)

    print(f"\nSaved summaries to: {csv_path}")
    print(f"Saved aggregate table to: {aggregate_path}")
    print(f"Saved paired tests to: {paired_path}")
    print(f"Results dir: {config.save_dir}")
    print(f"Time left (sec): {config.time_left_sec():.1f}")
    print_summary_table(all_summaries)


if __name__ == "__main__":
    main()
