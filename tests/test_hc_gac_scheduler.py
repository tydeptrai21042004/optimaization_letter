from __future__ import annotations

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import Controller, HCGACModulator
from lr_modulator.runtime import validate_method


def _config() -> ExperimentConfig:
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        eval_test_each_epoch=False,
    )
    cfg.mod_warmup_steps = 0
    cfg.m_win = 1
    cfg.trend_conf_tau = 0.0
    cfg.variance_normalize = True
    cfg.ema_gac_alignment_alpha = 0.0
    cfg.ema_gac_dead_zone = 0.0
    cfg.ema_gac_beta_up = 2.0
    cfg.ema_gac_beta_down = 2.0
    cfg.ema_gac_gamma_up = 0.02
    cfg.ema_gac_gamma_down = 0.06
    cfg.ema_gac_use_phase_envelope = False
    cfg.ema_gac_confirmation_mode = "loss_only"
    return cfg


def _make_modulator(cfg: ExperimentConfig):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    mod = HCGACModulator(
        optimizer=optimizer,
        config=cfg,
        base_mode="constant",
        total_steps=32,
        steps_per_epoch=8,
        base_lr=0.1,
        min_lr=1e-5,
    )
    return parameter, optimizer, mod


def test_hc_gac_is_registered_for_all_base_schedules() -> None:
    cfg = _config()
    for method in ["hc_gac_cosine", "hc_gac_onecycle", "hc_gac_warmup_cosine", "hc_gac_plateau"]:
        assert validate_method(method) == method
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        controller = Controller(
            optimizer=optimizer,
            config=cfg,
            method=method,
            total_steps=32,
            steps_per_epoch=8,
            base_lr=0.1,
            min_lr=1e-5,
        )
        assert controller.kind == "hc_gac"


def test_hc_gac_uses_delayed_convolution_and_keeps_positive_lr() -> None:
    cfg = _config()
    parameter, optimizer, mod = _make_modulator(cfg)

    for step in range(12):
        parameter.grad = torch.tensor([1.0])
        lr_before_gradient_observation = float(optimizer.param_groups[0]["lr"])
        mod.on_after_backward(0.0)
        assert float(optimizer.param_groups[0]["lr"]) == lr_before_gradient_observation
        mod.on_batch_end(1.0 / (step + 1))
        assert mod.hc.max_index_used_by_delayed_hc(mod.hc.batch_idx - 1) <= mod.hc.batch_idx - 1
        assert mod.last_mod_lr > 0.0
        assert -cfg.ema_gac_gamma_down <= mod.last_delta <= cfg.ema_gac_gamma_up

    assert mod.stats()["hc_kernel_l1"] > 0.0
