from __future__ import annotations

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import Controller


def test_controller_runs_onecycle_modulator() -> None:
    cfg = ExperimentConfig(use_amp=False, num_workers=0, do_finetune=False, warmup_steps=0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    controller = Controller(
        optimizer=optimizer,
        config=cfg,
        method="ours_onecycle",
        total_steps=8,
        steps_per_epoch=4,
        base_lr=0.1,
        min_lr=1e-4,
    )
    for loss in [1.2, 1.0, 0.9, 0.8]:
        controller.on_batch_end(loss)
    assert controller.last_lr > 0
