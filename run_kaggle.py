from __future__ import annotations

import os

from lr_modulator.config import ExperimentConfig
from lr_modulator.experiments import run_all
from lr_modulator.io_utils import save_aggregate_csv
from lr_modulator.runtime import get_device


def print_summary_table(all_summaries):
    print("\n" + "=" * 150)
    print(
        f"{'dataset':<12}{'model':<20}{'method':<16}{'seed':>4}"
        f"{'val':>10}{'test':>10}{'clip':>10}{'|d|':>10}{'beta_eff':>10}{'time(s)':>10}"
    )
    print("-" * 150)
    for s in all_summaries:
        clip = s.get("clip_rate", None)
        dabs = s.get("delta_mean_abs_final", None)
        beta_eff = s.get("beta_eff_mean", None)
        print(
            f"{s['dataset']:<12}{s['model']:<20}{s['method']:<16}{s['seed']:>4}"
            f"{s['best_val_acc']:>10.4f}{s['test_acc']:>10.4f}"
            f"{('-' if clip is None else f'{clip:.4f}'):>10}"
            f"{('-' if dabs is None else f'{dabs:.4f}'):>10}"
            f"{('-' if beta_eff is None else f'{beta_eff:.3f}'):>10}"
            f"{s['time_sec']:>10.1f}"
        )
    print("=" * 150)


def main() -> None:
    config = ExperimentConfig()
    device, _ = get_device()
    all_summaries = run_all(config, device)

    csv_path = os.path.join(config.save_dir, "all_run_summaries.csv")
    save_aggregate_csv(csv_path, all_summaries)

    print(f"\nSaved summaries to: {csv_path}")
    print(f"Results dir: {config.save_dir}")
    print(f"Time left (sec): {config.time_left_sec():.1f}")
    print_summary_table(all_summaries)


if __name__ == "__main__":
    main()
