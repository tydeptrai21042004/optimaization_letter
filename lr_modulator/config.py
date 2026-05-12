from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Top-level configuration for the LR-modulator project.

    Defaults are intentionally paper-oriented but still Kaggle-friendly.  The new
    fields support reviewer-requested rivals: random bounded modulation, L4,
    hypergradient SGD, D-Adaptation, and Prodigy, plus ablations for the proposed
    controller.
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
    # Reviewer-facing curves. If True, test loss/accuracy are logged at every epoch.
    # Disable for very large/time-constrained runs.
    eval_test_each_epoch: bool = True
    num_workers: int = 2
    download: bool = True

    # Grid.  Use >=5 seeds for paper results; use CLI --seeds for quick smoke runs.
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    scratch_datasets: List[str] = field(default_factory=lambda: ["cifar10", "cifar100"])
    scratch_models: List[str] = field(default_factory=lambda: ["resnet18"])
    scratch_methods: List[str] = field(
        default_factory=lambda: [
            # Existing schedules
            "cosine",
            "onecycle",
            "warmup_cosine",
            "warm_restarts",
            "plateau",
            # New direct controls / rivals
            "random_cosine",
            "random_onecycle",
            "l4_sgd",
            "hyper_sgd",
            "dadapt_sgd",
            "prodigy",
            "adamw",
            # Proposed method
            "ours_cosine",
            "ours_onecycle",
            "ours_warmup_cosine",
        ]
    )
    extra_baselines_cifar10: List[str] = field(default_factory=lambda: ["constant", "step"])

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
            "random_cosine",
            "random_onecycle",
            "l4_sgd",
            "hyper_sgd",
            "dadapt_sgd",
            "prodigy",
            "adamw",
            "ours_cosine",
            "ours_onecycle",
            "ours_warmup_cosine",
        ]
    )

    # Optional ablation method list for focused suite runs.
    ablation_methods: List[str] = field(
        default_factory=lambda: [
            "cosine",
            "random_cosine",
            "ours_no_ema_cosine",
            "ours_no_kernel_cosine",
            "ours_no_clip_cosine",
            "ours_deadzone_cosine",
            "ours_cosine",
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
    max_lr_factor: float = 10.0  # used by L4 / HyperSGD / fallback optimizers
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

    # Improved controller options.
    bias_correct_ema: bool = True
    relative_trend: bool = True
    dead_zone_tau: float = 0.0
    variance_normalize: bool = False
    var_alpha: float = 0.95
    eps_trend: float = 1e-8
    normalize_raw: bool = False

    # Ablation switches.  These can also be activated through method names:
    # ours_no_ema_cosine, ours_no_kernel_cosine, ours_no_clip_cosine.
    use_ema: bool = True
    use_kernel: bool = True
    use_clipping: bool = True
    emergency_delta_floor: float = -0.95  # keeps LR positive if clipping ablated
    emergency_delta_ceiling: float = 5.0

    use_auto_beta: bool = True
    beta_fixed: float = 0.08
    target_mean_abs_delta: float = 0.02
    beta_cap: float = 3.0

    # Random bounded modulation rival
    random_delta_gamma: float = 0.10

    # L4 rival.  Uses a practical loss-based step-size rule before optimizer.step().
    l4_alpha: float = 0.15
    l4_gamma: float = 0.90
    l4_eps: float = 1e-12

    # Hypergradient SGD rival.
    hyper_lr: float = 1e-6
    hyper_min_lr_factor: float = 0.01
    hyper_max_lr_factor: float = 10.0

    # D-Adaptation / Prodigy optimizer-level rivals.
    # If official packages are installed, they are used.  Otherwise the code uses
    # stable internal fallbacks and records this in the summary.
    dadapt_growth_rate: float = 1.02
    prodigy_beta1: float = 0.9
    prodigy_beta2: float = 0.999

    # Hyperparameter ablation grids for script-level sweeps.
    ablation_alphas: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95, 0.99])
    ablation_m_wins: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    ablation_rhos: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])
    ablation_betas: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.50])
    ablation_gammas: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])

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
