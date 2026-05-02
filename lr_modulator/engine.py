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
def eval_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
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
    epoch_index: int = 0,
) -> Tuple[float, float, List[Dict[str, float]]]:
    model.train()
    crit = nn.CrossEntropyLoss()
    use_amp = bool(config.use_amp and device.type == "cuda" and scaler.is_enabled())

    loss_sum = 0.0
    correct = 0
    total = 0
    batch_history: List[Dict[str, float]] = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))

        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with get_autocast_context(device, enabled=True):
                out = model(x)
                loss = crit(out, y)

            scaler.scale(loss).backward()
            # Unscale before L4 / HyperSGD so their gradient norms/dots use true gradients.
            scaler.unscale_(optimizer)
            controller.on_after_backward(loss.item())
            lr_after_backward = float(optimizer.param_groups[0]["lr"])
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            controller.on_after_backward(loss.item())
            lr_after_backward = float(optimizer.param_groups[0]["lr"])
            optimizer.step()

        # Batch-level controller update.  For the proposed method, this computes
        # delta_t from L_t and sets the LR to eta_{t+1}=r_{t+1}(1+delta_t).
        controller.on_batch_end(loss.item())
        lr_next = float(optimizer.param_groups[0]["lr"])

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

        batch_history.append(
            {
                "epoch": float(epoch_index + 1),
                "batch": float(batch_idx),
                "global_batch_in_epoch": float(batch_idx),
                "train_loss": float(loss.item()),
                "train_acc_batch": float((out.argmax(1) == y).float().mean().item()),
                "lr_used": lr_used,
                "lr_after_backward": lr_after_backward,
                "lr_next": lr_next,
                "delta": float(controller.last_delta),
                "raw": float(controller.last_raw),
            }
        )

        if config.should_stop():
            break

    return loss_sum / max(total, 1), correct / max(total, 1), batch_history


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
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[Dict[str, float]]]:
    scaler = make_grad_scaler(
        device,
        enabled=(config.use_amp and device.type == "cuda"),
    )

    best_val = -1.0
    best_state = None
    history: List[Dict[str, float]] = []
    batch_history_all: List[Dict[str, float]] = []

    for epoch in range(epochs):
        if config.should_stop():
            print("[STOP] Time budget reached mid-run (saving best so far).")
            break

        tr_loss, tr_acc, batch_history = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            controller=controller,
            scaler=scaler,
            device=device,
            config=config,
            epoch_index=epoch,
        )
        batch_history_all.extend(batch_history)

        va_loss, va_acc = eval_metrics(model, val_loader, device)

        # Epoch-level controller update.  Important for ReduceLROnPlateau.
        controller.on_epoch_end(va_loss)

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
            f"tr_acc={tr_acc:.4f} "
            f"va_acc={va_acc:.4f} "
            f"va_loss={va_loss:.4f} "
            f"lr={controller.last_lr:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc = eval_metrics(model, test_loader, device)

    final_metrics = {
        "best_val_acc": float(best_val),
        "test_acc": float(te_acc),
        "test_loss": float(te_loss),
    }

    final_metrics.update(controller.stats())

    return final_metrics, history, batch_history_all
