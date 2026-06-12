from __future__ import annotations

import math

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.optimizers import build_optimizer_for_method
from lr_modulator.runtime import validate_methods
from lr_modulator.schedulers import Controller


def _linear_optimizer_and_controller(method: str, total_steps: int = 12):
    torch.manual_seed(7)
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        do_finetune=False,
        mod_warmup_steps=0,
        m_win=1,
        gamma=0.05,
        use_auto_beta=False,
        beta_fixed=0.01,
        plateau_patience=1,
        plateau_factor=0.5,
    )
    model = torch.nn.Linear(4, 2)
    optimizer, _ = build_optimizer_for_method(method, model.parameters(), base_lr=0.1, config=cfg)
    controller = Controller(
        optimizer=optimizer,
        config=cfg,
        method=method,
        total_steps=total_steps,
        steps_per_epoch=4,
        base_lr=0.1,
        min_lr=1e-5,
    )
    return cfg, model, optimizer, controller


def test_default_proposal_methods_are_registered() -> None:
    methods = validate_methods(["plateau", "random_plateau", "ours_plateau", "ours_no_gate_plateau"])
    assert methods == ["plateau", "random_plateau", "ours_plateau", "ours_no_gate_plateau"]


def test_ours_plateau_is_no_gate_plateau_alias() -> None:
    _, _, _, alias = _linear_optimizer_and_controller("ours_plateau")
    _, _, _, explicit = _linear_optimizer_and_controller("ours_no_gate_plateau")

    assert alias.kind == "mod"
    assert explicit.kind == "mod"
    assert alias.mod.base.mode == "plateau"
    assert explicit.mod.base.mode == "plateau"
    assert alias.mod.variant == "no_gate"
    assert explicit.mod.variant == "no_gate"
    assert alias.mod._trend_conf_tau() == 0.0
    assert explicit.mod._trend_conf_tau() == 0.0


def test_ours_plateau_runs_batches_and_epoch_end_with_positive_lr() -> None:
    _, model, optimizer, controller = _linear_optimizer_and_controller("ours_plateau", total_steps=16)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for step in range(10):
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        controller.on_after_backward(float(loss.item()))
        optimizer.step()
        # Feed a smooth synthetic loss trend to make the no-gate HC modulator active
        # after its causality-safe warm-up.
        synthetic_loss = 1.0 - 0.02 * step
        losses.append(synthetic_loss)
        controller.on_batch_end(synthetic_loss)

    controller.on_epoch_end(0.9)
    stats = controller.stats()

    assert controller.last_lr > 0.0
    assert math.isfinite(controller.last_lr)
    assert stats["total_mod_steps"] > 0.0
    assert stats["active_mod_steps"] == stats["total_mod_steps"]
    assert 0.0 <= stats["delta_mean_abs_final"] <= 0.05 + 1e-12


def test_random_plateau_runs_batches_and_epoch_end_with_positive_lr() -> None:
    _, model, optimizer, controller = _linear_optimizer_and_controller("random_plateau", total_steps=8)
    criterion = torch.nn.CrossEntropyLoss()

    for step in range(3):
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        controller.on_batch_end(float(loss.item()))

    controller.on_epoch_end(1.0)
    assert controller.last_lr > 0.0
    assert abs(controller.last_delta) <= 0.05 + 1e-12
