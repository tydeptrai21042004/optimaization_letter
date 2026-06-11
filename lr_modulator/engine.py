from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .schedulers import Controller

TASK_TYPES = {"classification", "regression", "segmentation"}


def make_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def grad_norm_sq_from_optimizer(optimizer: torch.optim.Optimizer) -> float:
    total = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                g = p.grad.detach()
                total += float(torch.sum(g * g).item())
    return total


def _validate_task_type(task_type: str) -> str:
    task_type = task_type.lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type={task_type}. Supported: {sorted(TASK_TYPES)}")
    return task_type


def score_name_for_task(task_type: str) -> str:
    task_type = _validate_task_type(task_type)
    return {"classification": "accuracy", "segmentation": "pixel_accuracy", "regression": "rmse"}[task_type]


def lower_is_better(task_type: str) -> bool:
    return _validate_task_type(task_type) == "regression"


def make_criterion(task_type: str, config: ExperimentConfig) -> nn.Module:
    task_type = _validate_task_type(task_type)
    if task_type == "segmentation":
        return nn.CrossEntropyLoss(ignore_index=int(getattr(config, "segmentation_ignore_index", 255)))
    if task_type == "regression":
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def _model_output(output):
    return output["out"] if isinstance(output, dict) else output


def _prepare_target(y: torch.Tensor, out: torch.Tensor, task_type: str) -> torch.Tensor:
    task_type = _validate_task_type(task_type)
    if task_type == "regression":
        y = y.float()
        if y.ndim == 1:
            y = y.unsqueeze(1)
        if y.shape != out.shape:
            y = y.view_as(out)
        return y
    if task_type == "segmentation":
        if y.ndim == 4 and y.size(1) == 1:
            y = y[:, 0]
        return y.long()
    return y.long()


def _batch_metric(out: torch.Tensor, y: torch.Tensor, task_type: str) -> Tuple[float, int]:
    task_type = _validate_task_type(task_type)
    if task_type == "classification":
        return float((out.argmax(1) == y).sum().item()), int(y.numel())
    if task_type == "segmentation":
        pred = out.argmax(1)
        valid = y != 255
        denom = int(valid.sum().item())
        return (float((pred[valid] == y[valid]).sum().item()), denom) if denom else (0.0, 0)
    diff = out.detach().float() - y.detach().float()
    return float(torch.sum(diff * diff).item()), int(diff.numel())


def _final_score(metric_sum: float, metric_count: int, task_type: str) -> float:
    if metric_count <= 0:
        return float("nan")
    if _validate_task_type(task_type) == "regression":
        return float((metric_sum / metric_count) ** 0.5)
    return float(metric_sum / metric_count)


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device, task_type: str = "classification", config: ExperimentConfig | None = None) -> Tuple[float, float]:
    model.eval()
    task_type = _validate_task_type(task_type)
    if config is None:
        config = ExperimentConfig(use_amp=False, num_workers=0, download=False, do_finetune=False)
    crit = make_criterion(task_type, config)
    loss_sum, loss_count, metric_sum, metric_count = 0.0, 0, 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda")); y = y.to(device, non_blocking=(device.type == "cuda"))
        out = _model_output(model(x)); y_p = _prepare_target(y, out, task_type); loss = crit(out, y_p)
        loss_sum += loss.item() * x.size(0); loss_count += x.size(0)
        a, b = _batch_metric(out, y_p, task_type); metric_sum += a; metric_count += b
    return loss_sum / max(loss_count, 1), _final_score(metric_sum, metric_count, task_type)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, controller: Controller, scaler, device: torch.device, config: ExperimentConfig, epoch_index: int = 0, task_type: str = "classification") -> Tuple[float, float, List[Dict[str, float]]]:
    model.train()
    task_type = _validate_task_type(task_type); score_name = score_name_for_task(task_type); crit = make_criterion(task_type, config)
    use_amp = bool(config.use_amp and device.type == "cuda" and scaler.is_enabled())
    loss_sum, loss_count, metric_sum, metric_count = 0.0, 0, 0.0, 0
    batch_history: List[Dict[str, float]] = []
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=(device.type == "cuda")); y = y.to(device, non_blocking=(device.type == "cuda"))
        lr_used = float(optimizer.param_groups[0]["lr"]); optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with get_autocast_context(device, enabled=True):
                out = _model_output(model(x)); y_p = _prepare_target(y, out, task_type); loss = crit(out, y_p)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            controller.on_after_backward(loss.item()); grad_norm_sq = grad_norm_sq_from_optimizer(optimizer); controller.set_grad_norm_sq(grad_norm_sq)
            lr_after_backward = float(optimizer.param_groups[0]["lr"]); scaler.step(optimizer); scaler.update()
        else:
            out = _model_output(model(x)); y_p = _prepare_target(y, out, task_type); loss = crit(out, y_p)
            loss.backward(); controller.on_after_backward(loss.item()); grad_norm_sq = grad_norm_sq_from_optimizer(optimizer); controller.set_grad_norm_sq(grad_norm_sq)
            lr_after_backward = float(optimizer.param_groups[0]["lr"]); optimizer.step()
        controller.on_batch_end(loss.item()); lr_next = float(optimizer.param_groups[0]["lr"])
        loss_sum += loss.item() * x.size(0); loss_count += x.size(0)
        bm, bc = _batch_metric(out, y_p, task_type); metric_sum += bm; metric_count += bc; batch_score = _final_score(bm, bc, task_type)
        row = {"epoch": float(epoch_index + 1), "batch": float(batch_idx), "global_batch_in_epoch": float(batch_idx), "train_loss": float(loss.item()), "train_score_batch": float(batch_score), "train_acc_batch": float(batch_score) if task_type == "classification" else float("nan"), "lr_used": lr_used, "lr_after_backward": lr_after_backward, "lr_next": lr_next, "base_lr": float(controller.last_base_lr), "delta": float(controller.last_delta), "raw": float(controller.last_raw), "beta_eff": float(controller.last_beta_eff), "ema_loss": float(controller.last_ema_loss), "u_signal": float(controller.last_u_signal), "clipped": float(controller.last_clipped), "emergency_clipped": float(controller.last_emergency_clipped), "grad_norm_sq": float(controller.last_grad_norm_sq)}
        row[f"train_{score_name}_batch"] = float(batch_score); batch_history.append(row)
        if config.should_stop(): break
    return loss_sum / max(loss_count, 1), _final_score(metric_sum, metric_count, task_type), batch_history


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, optimizer: torch.optim.Optimizer, controller: Controller, device: torch.device, config: ExperimentConfig, epochs: int, task_type: str = "classification") -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, float]]]:
    task_type = _validate_task_type(task_type); score_name = score_name_for_task(task_type)
    scaler = make_grad_scaler(device, enabled=(config.use_amp and device.type == "cuda"))
    best_val_score = float("inf") if lower_is_better(task_type) else -float("inf"); best_val_loss = float("inf"); best_state = None
    history: List[Dict[str, float]] = []; batch_history_all: List[Dict[str, float]] = []
    for epoch in range(epochs):
        if config.should_stop():
            print("[STOP] Time budget reached mid-run (saving best so far)."); break
        tr_loss, tr_score, batch_history = train_one_epoch(model, train_loader, optimizer, controller, scaler, device, config, epoch, task_type)
        batch_history_all.extend(batch_history)
        va_loss, va_score = eval_metrics(model, val_loader, device, task_type, config)
        te_loss_epoch, te_score_epoch = eval_metrics(model, test_loader, device, task_type, config) if config.eval_test_each_epoch else (float("nan"), float("nan"))
        controller.on_epoch_end(va_loss)
        row = {"epoch": epoch + 1, "task_type": task_type, "score_name": score_name, "train_loss": float(tr_loss), "train_score": float(tr_score), "val_loss": float(va_loss), "val_score": float(va_score), "test_loss": float(te_loss_epoch), "test_score": float(te_score_epoch), "last_lr": float(controller.last_lr), "last_base_lr": float(controller.last_base_lr), "last_delta": float(controller.last_delta), "last_raw": float(controller.last_raw), "last_beta_eff": float(controller.last_beta_eff), "last_ema_loss": float(controller.last_ema_loss), "last_u_signal": float(controller.last_u_signal), "last_clip_flag": float(controller.last_clipped), "last_emergency_clip_flag": float(controller.last_emergency_clipped), "train_acc": float(tr_score) if task_type == "classification" else float("nan"), "val_acc": float(va_score) if task_type == "classification" else float("nan"), "test_acc": float(te_score_epoch) if task_type == "classification" else float("nan"), "generalization_gap": float(tr_score - va_score) if task_type == "classification" else float("nan")}
        row[f"train_{score_name}"] = float(tr_score); row[f"val_{score_name}"] = float(va_score); row[f"test_{score_name}"] = float(te_score_epoch); history.append(row)
        improved = va_score < best_val_score if lower_is_better(task_type) else va_score > best_val_score
        if improved:
            best_val_score, best_val_loss = va_score, va_loss; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[epoch {epoch + 1:03d}/{epochs:03d}] task={task_type} tr_{score_name}={tr_score:.4f} va_{score_name}={va_score:.4f} va_loss={va_loss:.4f} te_{score_name}={te_score_epoch:.4f} lr={controller.last_lr:.6f}")
    if best_state is not None: model.load_state_dict(best_state)
    te_loss, te_score = eval_metrics(model, test_loader, device, task_type, config)
    final_metrics = {"task_type": task_type, "score_name": score_name, "score_lower_is_better": float(1.0 if lower_is_better(task_type) else 0.0), "best_val_score": float(best_val_score), "best_val_loss": float(best_val_loss), "test_score": float(te_score), "test_loss": float(te_loss), f"best_val_{score_name}": float(best_val_score), f"test_{score_name}": float(te_score), "best_val_acc": float(best_val_score) if task_type == "classification" else float("nan"), "test_acc": float(te_score) if task_type == "classification" else float("nan")}
    final_metrics.update(controller.stats())
    return final_metrics, history, batch_history_all
