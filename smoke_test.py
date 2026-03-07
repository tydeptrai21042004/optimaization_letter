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
    tr, va, _ = random_split(ds, [40, 12, 12], generator=torch.Generator().manual_seed(7))
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(va, batch_size=batch_size, shuffle=False),
    )


def main() -> None:
    config = ExperimentConfig(
        global_walltime_hours=0.05,
        stop_grace_minutes=0,
        use_amp=False,
        num_workers=0,
        do_finetune=False,
        scratch_epochs=1,
        scratch_batch=8,
        warmup_steps=0,
    )
    device = torch.device("cpu")

    train_loader, val_loader = build_fake_loaders(batch_size=config.scratch_batch)
    model = TinyMLP(num_classes=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    controller = Controller(
        optimizer=optimizer,
        config=config,
        method="ours_cosine",
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

    payload = {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "last_lr": float(controller.last_lr),
        "stats": controller.stats(),
    }
    out_path = Path("smoke_test_output.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert controller.last_lr > 0, "learning rate must stay positive"
    assert 0.0 <= val_acc <= 1.0, "validation accuracy should be a probability"

    print("Smoke test passed.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
