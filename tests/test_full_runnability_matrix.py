from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lr_modulator.config import ExperimentConfig
from lr_modulator.engine import eval_metrics, fit, make_grad_scaler, train_one_epoch
from lr_modulator.optimizers import build_optimizer_for_method
from lr_modulator.schedulers import Controller


torch.set_num_threads(1)


def _cfg(**kwargs) -> ExperimentConfig:
    base = dict(
        global_walltime_hours=1.0,
        stop_grace_minutes=0,
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        skip_if_exists=False,
        eval_test_each_epoch=True,
        momentum=0.0,
        weight_decay=0.0,
        mod_warmup_steps=0,
        sched_warmup_steps=2,
        restart_t0_steps=2,
        restart_t_mult=1,
        plateau_patience=0,
        plateau_factor=0.5,
        random_delta_gamma=0.05,
        max_lr_factor=5.0,
    )
    base.update(kwargs)
    return ExperimentConfig(**base)


def _classification_loader(batch_size: int = 4) -> DataLoader:
    x = torch.linspace(-1.0, 1.0, steps=16 * 3 * 8 * 8).view(16, 3, 8, 8)
    y = torch.arange(16) % 3
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _regression_loader(batch_size: int = 4) -> DataLoader:
    x = torch.linspace(-1.0, 1.0, steps=16 * 3 * 8 * 8).view(16, 3, 8, 8)
    y = (x[:, 0].mean(dim=(1, 2)) - 0.5 * x[:, 1].mean(dim=(1, 2))).unsqueeze(1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _segmentation_loader(batch_size: int = 2) -> DataLoader:
    x = torch.zeros(8, 3, 8, 8)
    x[:, 0, :, 4:] = 1.0
    x[:, 1, 4:, :] = 1.0
    y = (x[:, 0] > 0.5).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


class TinyClassifier(nn.Module):
    def __init__(self, out_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinySegmenter(nn.Module):
    def __init__(self, out_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Conv2d(3, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_one_epoch(method: str, task_type: str) -> tuple[dict, list[dict]]:
    cfg = _cfg()
    device = torch.device("cpu")
    if task_type == "classification":
        model = TinyClassifier().to(device)
        loader = _classification_loader()
    elif task_type == "regression":
        model = TinyRegressor().to(device)
        loader = _regression_loader()
    else:
        model = TinySegmenter().to(device)
        loader = _segmentation_loader()

    opt, _ = build_optimizer_for_method(method, model.parameters(), base_lr=0.01, config=cfg)
    ctrl = Controller(opt, cfg, method, total_steps=len(loader), steps_per_epoch=len(loader), base_lr=0.01, min_lr=1e-5)
    scaler = make_grad_scaler(device, enabled=False)
    train_loss, train_score, hist = train_one_epoch(model, loader, opt, ctrl, scaler, device, cfg, epoch_index=0, task_type=task_type)
    val_loss, val_score = eval_metrics(model, loader, device, task_type=task_type, config=cfg)

    assert math.isfinite(train_loss)
    assert math.isfinite(train_score)
    assert math.isfinite(val_loss)
    assert math.isfinite(val_score)
    assert hist
    assert ctrl.last_lr > 0.0
    if task_type in {"classification", "segmentation"}:
        assert 0.0 <= train_score <= 1.0
        assert 0.0 <= val_score <= 1.0
    else:
        assert train_score >= 0.0
        assert val_score >= 0.0
    return {"last_lr": ctrl.last_lr, "last_delta": ctrl.last_delta, "stats": ctrl.stats()}, hist


def test_all_default_classification_baselines_and_proposed_methods_run_one_epoch() -> None:
    methods = [
        "constant",
        "step",
        "cosine",
        "onecycle",
        "warmup_cosine",
        "warm_restarts",
        "plateau",
        "random_cosine",
        "random_onecycle",
        "random_warmup_cosine",
        "l4_sgd",
        "hyper_sgd",
        "dadapt_sgd",
        "prodigy",
        "adamw",
        "ours_cosine",
        "ours_onecycle",
        "ours_warmup_cosine",
        "ours_no_hc_cosine",
        "ours_no_noise_norm_cosine",
        "ours_no_gate_cosine",
        "ours_no_phi_cosine",
        "ours_no_clip_cosine",
    ]
    for method in methods:
        payload, hist = _run_one_epoch(method, "classification")
        assert payload["last_lr"] > 0.0
        assert "train_loss" in hist[0]


def test_regression_and_segmentation_methods_run_one_epoch() -> None:
    for task_type, methods in {
        "regression": ["constant", "cosine", "random_cosine", "l4_sgd", "hyper_sgd", "ours_cosine"],
        "segmentation": ["constant", "cosine", "random_cosine", "ours_cosine"],
    }.items():
        for method in methods:
            payload, hist = _run_one_epoch(method, task_type)
            assert payload["last_lr"] > 0.0
            assert hist[0]["train_loss"] >= 0.0


def test_fit_returns_task_aware_summary_fields_for_classification_regression_segmentation() -> None:
    cfg = _cfg(eval_test_each_epoch=True)
    device = torch.device("cpu")
    cases = [
        ("classification", TinyClassifier(), _classification_loader()),
        ("regression", TinyRegressor(), _regression_loader()),
        ("segmentation", TinySegmenter(), _segmentation_loader()),
    ]
    for task_type, model, loader in cases:
        opt, _ = build_optimizer_for_method("cosine", model.parameters(), base_lr=0.01, config=cfg)
        ctrl = Controller(opt, cfg, "cosine", total_steps=len(loader), steps_per_epoch=len(loader), base_lr=0.01, min_lr=1e-5)
        final, history, batch_history = fit(model, loader, loader, loader, opt, ctrl, device, cfg, epochs=1, task_type=task_type)
        assert final["task_type"] == task_type
        assert "best_val_score" in final and "test_score" in final
        assert "score_name" in final
        assert history and batch_history
        assert math.isfinite(final["test_score"])
        assert math.isfinite(final["test_loss"])
