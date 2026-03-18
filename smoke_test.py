from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from lr_modulator.config import ExperimentConfig
from lr_modulator.engine import eval_metrics, make_grad_scaler, train_one_epoch
from lr_modulator.schedulers import Controller


class TinyMLP(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 16 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_fake_loaders(batch_size: int = 8):
    torch.manual_seed(7)
    x = torch.randn(64, 3, 16, 16)
    y = torch.randint(0, 3, (64,))
    ds = TensorDataset(x, y)
    tr, va, _ = random_split(
        ds,
        [40, 12, 12],
        generator=torch.Generator().manual_seed(7),
    )
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
) -> dict:
    model = TinyMLP(num_classes=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)

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

    train_loss, train_acc = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        controller=controller,
        scaler=scaler,
        device=device,
        config=config,
    )

    val_loss, val_acc = eval_metrics(model, val_loader, device)

    # Important for plateau and safe for the other methods.
    controller.on_epoch_end(val_loss)

    payload = {
        "method": method,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "last_lr": float(controller.last_lr),
        "last_delta": float(controller.last_delta),
        "last_raw": float(controller.last_raw),
        "stats": controller.stats(),
    }

    assert controller.last_lr > 0.0, f"{method}: learning rate must stay positive"
    assert 0.0 <= val_acc <= 1.0, f"{method}: validation accuracy should be a probability"

    return payload


def main() -> None:
    config = ExperimentConfig(
        global_walltime_hours=0.05,
        stop_grace_minutes=0,
        use_amp=False,
        num_workers=0,
        do_finetune=False,
        scratch_epochs=1,
        scratch_batch=8,
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

    methods = [
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

    results = []
    for method in methods:
        result = run_one_method(
            method=method,
            config=config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        results.append(result)

    payload = {
        "num_methods_tested": len(results),
        "methods": results,
    }

    out_path = Path("smoke_test_output.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Smoke test passed.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
