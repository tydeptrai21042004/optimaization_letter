from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lr_modulator.config import ExperimentConfig
from lr_modulator.data import DATASET_INFO, build_loaders, recommended_input_size, task_type_for_dataset
from lr_modulator.engine import eval_metrics, make_grad_scaler, train_one_epoch
from lr_modulator.model_zoo import build_model
from lr_modulator.optimizers import build_optimizer_for_method
from lr_modulator.schedulers import Controller


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        global_walltime_hours=1.0,
        stop_grace_minutes=0,
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        skip_if_exists=False,
        mod_warmup_steps=0,
        sched_warmup_steps=2,
    )


def _controller(model: nn.Module, cfg: ExperimentConfig, steps: int = 2) -> tuple[torch.optim.Optimizer, Controller]:
    opt, _ = build_optimizer_for_method("ours_cosine", model.parameters(), base_lr=0.01, config=cfg)
    ctrl = Controller(opt, cfg, "ours_cosine", total_steps=steps, steps_per_epoch=max(steps, 1), base_lr=0.01, min_lr=1e-5)
    return opt, ctrl


def test_dataset_registry_contains_multiple_task_types() -> None:
    assert task_type_for_dataset("cifar10") == "classification"
    assert task_type_for_dataset("synthetic_regression") == "regression"
    assert task_type_for_dataset("synthetic_segmentation") == "segmentation"
    assert "stl10" in DATASET_INFO
    assert "voc_segmentation" in DATASET_INFO


def test_synthetic_regression_loader_and_tiny_model_train() -> None:
    cfg = _cfg()
    device = torch.device("cpu")
    input_size = recommended_input_size("tiny_cnn", "synthetic_regression", pretrained=False)
    tr, va, _, out_dim, task_type = build_loaders(cfg, device, "synthetic_regression", input_size, batch_size=8, seed=0)
    model = build_model("tiny_cnn", out_dim, pretrained=False, input_size=input_size, task_type=task_type)
    opt, ctrl = _controller(model, cfg, steps=len(tr))
    scaler = make_grad_scaler(device, enabled=False)
    train_loss, train_rmse, hist = train_one_epoch(model, tr, opt, ctrl, scaler, device, cfg, task_type=task_type)
    val_loss, val_rmse = eval_metrics(model, va, device, task_type=task_type, config=cfg)
    assert train_loss >= 0.0 and val_loss >= 0.0
    assert train_rmse >= 0.0 and val_rmse >= 0.0
    assert hist and "train_rmse_batch" in hist[0]


def test_synthetic_segmentation_loader_and_tiny_unet_train() -> None:
    cfg = _cfg()
    device = torch.device("cpu")
    input_size = recommended_input_size("tiny_unet", "synthetic_segmentation", pretrained=False)
    tr, va, _, out_dim, task_type = build_loaders(cfg, device, "synthetic_segmentation", input_size, batch_size=4, seed=0)
    model = build_model("tiny_unet", out_dim, pretrained=False, input_size=input_size, task_type=task_type)
    opt, ctrl = _controller(model, cfg, steps=len(tr))
    scaler = make_grad_scaler(device, enabled=False)
    train_loss, train_pixel_acc, hist = train_one_epoch(model, tr, opt, ctrl, scaler, device, cfg, task_type=task_type)
    val_loss, val_pixel_acc = eval_metrics(model, va, device, task_type=task_type, config=cfg)
    assert train_loss >= 0.0 and val_loss >= 0.0
    assert 0.0 <= train_pixel_acc <= 1.0
    assert 0.0 <= val_pixel_acc <= 1.0
    assert hist and "train_pixel_accuracy_batch" in hist[0]
