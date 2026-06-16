from __future__ import annotations

import math

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import Controller, EMAGACModulator
from lr_modulator.runtime import validate_method


def _config() -> ExperimentConfig:
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        eval_test_each_epoch=False,
    )
    cfg.ema_gac_alpha_fast = 0.0
    cfg.ema_gac_alpha_slow = 0.5
    cfg.ema_gac_volatility_alpha = 0.0
    cfg.ema_gac_alignment_alpha = 0.0
    cfg.ema_gac_dead_zone = 0.0
    cfg.ema_gac_beta_up = 2.0
    cfg.ema_gac_beta_down = 2.0
    cfg.ema_gac_gamma_up = 0.02
    cfg.ema_gac_gamma_down = 0.06
    cfg.ema_gac_use_phase_envelope = False
    cfg.ema_gac_confirmation_mode = "strict"
    return cfg


def _make_modulator(cfg: ExperimentConfig):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    mod = EMAGACModulator(
        optimizer=optimizer,
        config=cfg,
        base_mode="constant",
        total_steps=20,
        steps_per_epoch=10,
        base_lr=0.1,
        min_lr=1e-5,
    )
    return parameter, optimizer, mod


def _backward_with_gradient(parameter: torch.nn.Parameter, mod: EMAGACModulator, value: float) -> None:
    parameter.grad = torch.tensor([value])
    mod.on_after_backward(0.0)


def test_ema_gac_increases_only_when_improvement_is_gradient_confirmed() -> None:
    cfg = _config()
    parameter, _, mod = _make_modulator(cfg)

    _backward_with_gradient(parameter, mod, 1.0)
    mod.on_batch_end(1.0)
    assert mod.last_delta == 0.0

    _backward_with_gradient(parameter, mod, 1.0)  # positive alignment
    mod.on_batch_end(0.5)  # improving loss: slow EMA > fast EMA
    assert mod.last_alignment > 0.99
    assert mod.last_raw > 0.0
    assert 0.0 < mod.last_delta <= cfg.ema_gac_gamma_up


def test_ema_gac_decreases_when_worsening_and_misalignment_agree() -> None:
    cfg = _config()
    parameter, _, mod = _make_modulator(cfg)

    _backward_with_gradient(parameter, mod, 1.0)
    mod.on_batch_end(1.0)
    _backward_with_gradient(parameter, mod, -1.0)  # negative alignment
    mod.on_batch_end(2.0)  # worsening loss: slow EMA < fast EMA

    assert mod.last_alignment < -0.99
    assert mod.last_raw < 0.0
    assert -cfg.ema_gac_gamma_down <= mod.last_delta < 0.0


def test_ema_gac_disagreement_preserves_base_schedule() -> None:
    cfg = _config()
    parameter, _, mod = _make_modulator(cfg)

    _backward_with_gradient(parameter, mod, 1.0)
    mod.on_batch_end(1.0)
    _backward_with_gradient(parameter, mod, -1.0)  # misaligned
    mod.on_batch_end(0.5)  # improving, so signs disagree

    assert mod.last_raw > 0.0
    assert mod.last_alignment < 0.0
    assert mod.last_confirmed_signal == 0.0
    assert mod.last_delta == 0.0
    assert math.isclose(mod.last_mod_lr, mod.last_base_lr, rel_tol=0.0, abs_tol=1e-12)


def test_ema_gac_is_next_step_causal_and_asymmetrically_bounded() -> None:
    cfg = _config()
    parameter, optimizer, mod = _make_modulator(cfg)
    lr_before = optimizer.param_groups[0]["lr"]

    _backward_with_gradient(parameter, mod, 1.0)
    # Gradient observation alone must not alter the LR used by this optimizer step.
    assert optimizer.param_groups[0]["lr"] == lr_before
    mod.on_batch_end(1.0)

    for gradient, loss in [(1.0, 0.1), (-1.0, 10.0), (1.0, 0.01)]:
        lr_used = optimizer.param_groups[0]["lr"]
        _backward_with_gradient(parameter, mod, gradient)
        assert optimizer.param_groups[0]["lr"] == lr_used
        mod.on_batch_end(loss)
        assert -cfg.ema_gac_gamma_down <= mod.last_delta <= cfg.ema_gac_gamma_up
        assert mod.last_mod_lr > 0.0


def test_controller_and_runtime_register_all_ema_gac_bases() -> None:
    cfg = _config()
    for method in [
        "ema_gac_cosine",
        "ema_gac_onecycle",
        "ema_gac_warmup_cosine",
        "ema_gac_plateau",
    ]:
        assert validate_method(method) == method
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        controller = Controller(
            optimizer=optimizer,
            config=cfg,
            method=method,
            total_steps=20,
            steps_per_epoch=10,
            base_lr=0.1,
            min_lr=1e-5,
        )
        assert controller.kind == "ema_gac"
