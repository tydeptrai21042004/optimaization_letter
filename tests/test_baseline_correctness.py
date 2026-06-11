from __future__ import annotations

import math

import numpy as np
import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.optimizers import build_optimizer_for_method
from lr_modulator.schedulers import BatchBaseSchedule, Controller, HyperGradientController, L4StepController


def _cfg(**kwargs) -> ExperimentConfig:
    base = dict(
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        deterministic=True,
        skip_if_exists=False,
        momentum=0.0,
        weight_decay=0.0,
        min_lr=1e-4,
        step_size_epochs=2,
        step_gamma=0.5,
        sched_warmup_steps=3,
        warmup_start_factor=0.2,
        plateau_patience=0,
        plateau_factor=0.5,
        plateau_threshold=0.0,
        random_delta_gamma=0.07,
        max_lr_factor=10.0,
        l4_alpha=0.15,
        l4_gamma=0.90,
        hyper_lr=1e-3,
    )
    base.update(kwargs)
    return ExperimentConfig(**base)


def _optimizer(lr: float = 0.1) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    return torch.optim.SGD([param], lr=lr)


def test_constant_step_cosine_warmup_schedules_match_closed_form() -> None:
    cfg = _cfg()
    base_lr, min_lr, total_steps, steps_per_epoch = 0.1, 1e-4, 20, 5

    constant = BatchBaseSchedule(_optimizer(base_lr), cfg, "constant", total_steps, steps_per_epoch, base_lr, min_lr)
    assert constant.lr_at(0) == base_lr
    assert constant.lr_at(7) == base_lr
    assert constant.lr_at(total_steps) == base_lr

    step = BatchBaseSchedule(_optimizer(base_lr), cfg, "step", total_steps, steps_per_epoch, base_lr, min_lr)
    assert step.lr_at(0) == base_lr
    assert step.lr_at(9) == base_lr
    assert math.isclose(step.lr_at(10), base_lr * cfg.step_gamma, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(step.lr_at(20), base_lr * (cfg.step_gamma ** 2), rel_tol=0.0, abs_tol=1e-12)

    cosine = BatchBaseSchedule(_optimizer(base_lr), cfg, "cosine", total_steps, steps_per_epoch, base_lr, min_lr)
    assert math.isclose(cosine.lr_at(0), base_lr, rel_tol=0.0, abs_tol=1e-12)
    mid_expected = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * 10 / total_steps))
    assert math.isclose(cosine.lr_at(10), mid_expected, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(cosine.lr_at(total_steps), min_lr, rel_tol=0.0, abs_tol=1e-12)

    warmup = BatchBaseSchedule(_optimizer(base_lr), cfg, "warmup_cosine", total_steps, steps_per_epoch, base_lr, min_lr)
    assert math.isclose(warmup.lr_at(0), base_lr * cfg.warmup_start_factor, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(warmup.lr_at(cfg.sched_warmup_steps), base_lr, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(warmup.lr_at(total_steps), min_lr, rel_tol=0.0, abs_tol=1e-12)


def test_plateau_baseline_reduces_only_on_epoch_metric() -> None:
    cfg = _cfg()
    opt = _optimizer(0.1)
    sched = BatchBaseSchedule(opt, cfg, "plateau", total_steps=5, steps_per_epoch=5, base_lr=0.1, min_lr=1e-4)
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, rel_tol=0.0, abs_tol=1e-12)

    sched.on_batch_end()
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, rel_tol=0.0, abs_tol=1e-12)

    sched.on_epoch_end(1.0)  # establishes best metric
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, rel_tol=0.0, abs_tol=1e-12)

    sched.on_epoch_end(1.1)  # worse metric, patience=0 => reduce immediately
    assert math.isclose(opt.param_groups[0]["lr"], 0.05, rel_tol=0.0, abs_tol=1e-12)


def _lr_sequence_for_losses(method: str, losses: list[float]) -> list[float]:
    cfg = _cfg()
    opt = _optimizer(0.1)
    ctrl = Controller(opt, cfg, method, total_steps=len(losses), steps_per_epoch=max(1, len(losses)), base_lr=0.1, min_lr=1e-4)
    lrs = []
    for loss in losses:
        ctrl.on_batch_end(loss)
        lrs.append(float(opt.param_groups[0]["lr"]))
    return lrs


def test_standard_baselines_are_loss_independent_and_have_zero_feedback_delta() -> None:
    low_losses = [0.1, 0.2, 0.15, 0.12]
    high_losses = [5.0, 1.0, 4.0, 3.0]
    for method in ["constant", "step", "cosine", "warmup_cosine"]:
        assert _lr_sequence_for_losses(method, low_losses) == _lr_sequence_for_losses(method, high_losses)

        cfg = _cfg()
        opt = _optimizer(0.1)
        ctrl = Controller(opt, cfg, method, total_steps=4, steps_per_epoch=4, base_lr=0.1, min_lr=1e-4)
        for loss in high_losses:
            ctrl.on_batch_end(loss)
            assert ctrl.last_delta == 0.0
            assert ctrl.last_raw == 0.0
            assert ctrl.stats() == {}


def test_random_baseline_is_bounded_around_its_base_schedule() -> None:
    np.random.seed(123)
    cfg = _cfg(random_delta_gamma=0.07)
    opt = _optimizer(0.1)
    ctrl = Controller(opt, cfg, "random_cosine", total_steps=8, steps_per_epoch=8, base_lr=0.1, min_lr=1e-4)
    for loss in [1.0, 0.9, 1.1, 0.8]:
        ctrl.on_batch_end(loss)
        assert abs(ctrl.last_delta) <= cfg.random_delta_gamma + 1e-12
        expected_lr = ctrl.last_base_lr * (1.0 + ctrl.last_delta)
        assert math.isclose(ctrl.last_lr, expected_lr, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(opt.param_groups[0]["lr"], expected_lr, rel_tol=0.0, abs_tol=1e-12)
    assert ctrl.stats()["random_delta_mean_abs"] > 0.0


def test_l4_closed_form_first_update() -> None:
    cfg = _cfg(l4_alpha=0.15, l4_gamma=0.90, min_lr=1e-4, max_lr_factor=10.0)
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    ctrl = L4StepController(opt, cfg, base_lr=0.1, min_lr=cfg.min_lr)

    loss = (p - 3.0).pow(2).sum()  # loss=4, grad=-4, ||g||^2=16
    loss.backward()
    ctrl.on_after_backward(float(loss.item()))

    expected = cfg.l4_alpha * (float(loss.item()) - cfg.l4_gamma * float(loss.item())) / 16.0
    assert math.isclose(ctrl.last_lr, expected, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(opt.param_groups[0]["lr"], expected, rel_tol=1e-6, abs_tol=1e-12)
    assert ctrl.stats()["l4_updates"] == 1.0


def test_hypergradient_closed_form_second_update() -> None:
    cfg = _cfg(hyper_lr=1e-3, min_lr=1e-4, hyper_min_lr_factor=0.01, hyper_max_lr_factor=10.0)
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=0.1)
    ctrl = HyperGradientController(opt, cfg, base_lr=0.1, min_lr=cfg.min_lr)

    p.grad = torch.tensor([2.0, -1.0])
    ctrl.on_after_backward(1.0)
    assert math.isclose(ctrl.last_lr, 0.1, rel_tol=0.0, abs_tol=1e-12)
    ctrl.on_batch_end(1.0)

    p.grad = torch.tensor([3.0, 4.0])
    ctrl.on_after_backward(1.0)
    expected = 0.1 + cfg.hyper_lr * float(2.0 * 3.0 + (-1.0) * 4.0)
    assert math.isclose(ctrl.last_lr, expected, rel_tol=1e-6, abs_tol=1e-12)
    assert math.isclose(opt.param_groups[0]["lr"], expected, rel_tol=1e-6, abs_tol=1e-12)
    assert ctrl.stats()["hyper_updates"] == 2.0


def test_optimizer_only_baselines_use_correct_optimizer_classes_or_recorded_fallbacks() -> None:
    cfg = _cfg()
    for method in ["adamw", "dadapt_sgd", "prodigy"]:
        model = torch.nn.Linear(4, 2)
        opt, impl = build_optimizer_for_method(method, model.parameters(), base_lr=0.01, config=cfg)
        ctrl = Controller(opt, cfg, method, total_steps=2, steps_per_epoch=2, base_lr=0.01, min_lr=1e-4)
        assert ctrl.kind == "optimizer_only"
        assert ctrl.last_delta == 0.0
        assert impl in {"torch_adamw", "official_dadaptation", "internal_fallback", "official_prodigyopt"}
        if method == "adamw":
            assert isinstance(opt, torch.optim.AdamW)
            assert impl == "torch_adamw"
