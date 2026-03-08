from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .schedulers import Controller


def make_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except Exception:  # pragma: no cover - compatibility fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def get_autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except Exception:  # pragma: no cover - compatibility fallback
        return torch.cuda.amp.autocast(enabled=True)


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        out = model(x)
        loss = crit(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    controller: Controller,
    scaler,
    device: torch.device,
    config: ExperimentConfig,
) -> Tuple[float, float]:
    model.train()
    crit = nn.CrossEntropyLoss()
    use_amp = bool(config.use_amp and device.type == "cuda" and scaler.is_enabled())

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with get_autocast_context(device, enabled=True):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optimizer.step()

        controller.on_batch_end(loss.item())

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

        if config.should_stop():
            break

    return loss_sum / max(total, 1), correct / max(total, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    controller: Controller,
    device: torch.device,
    config: ExperimentConfig,
    epochs: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    scaler = make_grad_scaler(device, enabled=(config.use_amp and device.type == "cuda"))
    best_val = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        if config.should_stop():
            print("[STOP] Time budget reached mid-run (saving best so far).")
            break

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, controller, scaler, device, config)
        va_loss, va_acc = eval_metrics(model, val_loader, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
            "last_lr": float(controller.last_lr),
            "last_delta": float(controller.last_delta),
            "last_raw": float(controller.last_raw),
        }
        history.append(row)

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"[epoch {epoch + 1:03d}/{epochs:03d}] "
            f"tr_acc={tr_acc:.4f} va_acc={va_acc:.4f} lr={controller.last_lr:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc = eval_metrics(model, test_loader, device)
    final_metrics = {
        "best_val_acc": float(best_val),
        "test_acc": float(te_acc),
        "test_loss": float(te_loss),
    }
    return final_metrics, history