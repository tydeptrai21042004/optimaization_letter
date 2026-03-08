from __future__ import annotations

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import Controller


def test_controller_runs_onecycle_modulator() -> None:
    torch.manual_seed(0)

    cfg = ExperimentConfig(use_amp=False, num_workers=0, do_finetune=False, warmup_steps=0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    controller = Controller(
        optimizer=optimizer,
        config=cfg,
        method="ours_onecycle",
        total_steps=8,
        steps_per_epoch=4,
        base_lr=0.1,
        min_lr=1e-4,
    )

    for _ in range(4):
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        controller.on_batch_end(loss.item())

    assert controller.last_lr > 0.0