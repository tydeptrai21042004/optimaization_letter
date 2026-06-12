from __future__ import annotations

import json
import os
from pathlib import Path

import torch

torch.set_num_threads(1)

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from lr_modulator.config import ExperimentConfig
from lr_modulator.engine import eval_metrics, make_grad_scaler, train_one_epoch
from lr_modulator.optimizers import build_optimizer_for_method
from lr_modulator.schedulers import Controller


class TinyMLP(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_fake_loaders(batch_size: int = 8):
    torch.manual_seed(7)
    x = torch.randn(12, 3, 16, 16)
    y = torch.randint(0, 3, (12,))
    ds = TensorDataset(x, y)
    tr, va, _ = random_split(ds, [8, 2, 2], generator=torch.Generator().manual_seed(7))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(va, batch_size=batch_size, shuffle=False),
    )


def build_fake_regression_loaders(batch_size: int = 8):
    torch.manual_seed(11)
    x = torch.randn(12, 3, 16, 16)
    y = (x[:, 0].mean(dim=(1, 2)) - 0.5 * x[:, 1].mean(dim=(1, 2))).unsqueeze(1)
    ds = TensorDataset(x, y)
    tr, va, _ = random_split(ds, [8, 2, 2], generator=torch.Generator().manual_seed(11))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(va, batch_size=batch_size, shuffle=False),
    )


def run_one_method(
    method: str,
    config: ExperimentConfig,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_type: str = "classification",
) -> dict:
    model = (TinyRegressor() if task_type == "regression" else TinyMLP(num_classes=3)).to(device)
    optimizer, optimizer_impl = build_optimizer_for_method(method, model.parameters(), base_lr=0.05, config=config)

    controller = Controller(
        optimizer=optimizer,
        config=config,
        method=method,
        total_steps=len(train_loader),
        steps_per_epoch=len(train_loader),
        base_lr=0.05,
        min_lr=1e-4,
    )

    scaler = make_grad_scaler(device, enabled=False)

    train_loss, train_score, batch_history = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        controller=controller,
        scaler=scaler,
        device=device,
        config=config,
        epoch_index=0,
        task_type=task_type,
    )

    val_loss, val_score = eval_metrics(model, val_loader, device, task_type=task_type, config=config)
    score_name = "rmse" if task_type == "regression" else "accuracy"
    controller.on_epoch_end(val_loss)

    payload = {
        "method": method,
        "task_type": task_type,
        "optimizer_impl": optimizer_impl,
        "score_name": score_name,
        "train_loss": float(train_loss),
        "train_score": float(train_score),
        "val_loss": float(val_loss),
        "val_score": float(val_score),
        "train_acc": float(train_score) if task_type == "classification" else float("nan"),
        "val_acc": float(val_score) if task_type == "classification" else float("nan"),
        "last_lr": float(controller.last_lr),
        "last_delta": float(controller.last_delta),
        "last_raw": float(controller.last_raw),
        "num_batch_rows": len(batch_history),
        "stats": controller.stats(),
    }

    assert controller.last_lr > 0.0, f"{method}: learning rate must stay positive"
    if task_type == "classification":
        assert 0.0 <= val_score <= 1.0, f"{method}: validation accuracy should be a probability"
    else:
        assert val_score >= 0.0, f"{method}: regression RMSE should be non-negative"
    assert len(batch_history) > 0, f"{method}: batch history should be recorded"

    return payload


def main() -> None:
    config = ExperimentConfig(
        global_walltime_hours=0.05,
        stop_grace_minutes=0,
        use_amp=False,
        num_workers=0,
        do_finetune=False,
        scratch_epochs=1,
        scratch_batch=4,
        mod_warmup_steps=0,
        sched_warmup_steps=2,
        warmup_start_factor=0.2,
        restart_t0_steps=2,
        restart_t_mult=2,
        plateau_patience=1,
        plateau_factor=0.5,
    )
    device = torch.device("cpu")

    train_loader, val_loader = build_fake_loaders(batch_size=config.scratch_batch)
    reg_train_loader, reg_val_loader = build_fake_regression_loaders(batch_size=config.scratch_batch)

    methods = [
        "constant",
        "plateau",
        "random_plateau",
        "ours_plateau",
        "ours_no_gate_plateau",
        "random_cosine",
        "ours_cosine",
    ]

    results = []
    for method in methods:
        print(f"[smoke] running classification {method}", flush=True)
        result = run_one_method(method, config, device, train_loader, val_loader, task_type="classification")
        results.append(result)

    print("[smoke] running regression ours_plateau", flush=True)
    results.append(run_one_method("ours_plateau", config, device, reg_train_loader, reg_val_loader, task_type="regression"))

    payload = {"num_methods_tested": len(results), "methods": results}

    out_path = Path("smoke_test_output.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Smoke test passed.")
    print(json.dumps(payload, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
