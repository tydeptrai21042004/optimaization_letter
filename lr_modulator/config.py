from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Top-level configuration for the project.

    The defaults mirror a paper-minimum Kaggle-friendly grid.
    Edit this file if you want to expand the run plan.
    """

    # Paths
    is_kaggle: bool = field(
        default_factory=lambda: os.path.exists("/kaggle")
        and os.path.exists("/kaggle/working")
    )
    data_root: str = field(init=False)
    save_dir: str = field(init=False)

    # Time budget / skip behavior
    global_walltime_hours: float = 11.3
    stop_grace_minutes: int = 10
    start_time: float = field(default_factory=time.time)
    skip_if_exists: bool = True
    skip_on_fail: bool = True

    # Reproducibility / runtime
    deterministic: bool = True
    use_amp: bool = True
    num_workers: int = 2
    download: bool = True

    # Grid
    seeds: List[int] = field(default_factory=lambda: [0])

    scratch_datasets: List[str] = field(default_factory=lambda: ["cifar10", "cifar100"])
    scratch_models: List[str] = field(default_factory=lambda: ["resnet18"])
    scratch_methods: List[str] = field(
        default_factory=lambda: [
            "cosine",
            "onecycle",
            "warmup_cosine",
            "warm_restarts",
            "plateau",
            "ours_cosine",
            "ours_onecycle",
            "ours_warmup_cosine",
        ]
    )
    extra_baselines_cifar10: List[str] = field(
        default_factory=lambda: ["constant", "step"]
    )

    do_finetune: bool = True
    finetune_datasets: List[str] = field(default_factory=lambda: ["oxfordiiitpet"])
    finetune_models: List[str] = field(default_factory=lambda: ["resnet50"])
    finetune_methods: List[str] = field(
        default_factory=lambda: [
            "constant",
            "step",
            "cosine",
            "onecycle",
            "warmup_cosine",
            "warm_restarts",
            "plateau",
            "ours_cosine",
            "ours_onecycle",
            "ours_warmup_cosine",
        ]
    )

    # Optimization
    scratch_epochs: int = 40
    finetune_epochs: int = 10
    scratch_batch: int = 128
    finetune_batch: int = 64

    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scratch: float = 0.1
    lr_finetune: float = 0.01
    min_lr: float = 1e-4
    val_ratio: float = 0.1

    # Step baseline
    step_size_epochs: int = 20
    step_gamma: float = 0.1

    # EMA-loss modulator hyperparameters
    alpha: float = 0.95
    c_phi: float = 1.0
    m_win: int = 3
    rho: float = 0.8
    gamma: float = 0.10
    normalize_raw: bool = False

    use_auto_beta: bool = True
    beta_fixed: float = 0.08
    target_mean_abs_delta: float = 0.02
    beta_cap: float = 3.0

    # Separate warmups
    # For the EMA modulator activation
    mod_warmup_steps: int = 100

    # For the warmup+cosine base schedule
    sched_warmup_steps: int = 500
    warmup_start_factor: float = 0.1

    # Cosine warm restarts
    restart_t0_steps: int = 1000
    restart_t_mult: int = 2

    # Plateau scheduler
    plateau_mode: str = "min"
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_threshold: float = 1e-4
    plateau_threshold_mode: str = "rel"
    plateau_cooldown: int = 0

    def __post_init__(self) -> None:
        self.data_root = "/kaggle/working/data" if self.is_kaggle else "./data"
        self.save_dir = (
            "/kaggle/working/results_lr_modulator"
            if self.is_kaggle
            else "./results_lr_modulator"
        )
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

    def time_left_sec(self) -> float:
        return self.start_time + self.global_walltime_hours * 3600.0 - time.time()

    def should_stop(self) -> bool:
        return self.time_left_sec() <= self.stop_grace_minutes * 60.0
