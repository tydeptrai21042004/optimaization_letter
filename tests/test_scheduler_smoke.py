from __future__ import annotations

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.optimizers import build_optimizer_for_method
from lr_modulator.schedulers import Controller


def _run_controller_method(method: str, total_steps: int = 8) -> Controller:
    torch.manual_seed(0)
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        do_finetune=False,
        mod_warmup_steps=0,
        sched_warmup_steps=2,
        restart_t0_steps=2,
        restart_t_mult=2,
        plateau_patience=1,
    )
    model = torch.nn.Linear(4, 2)
    optimizer, _ = build_optimizer_for_method(method, model.parameters(), base_lr=0.1, config=cfg)
    criterion = torch.nn.CrossEntropyLoss()

    controller = Controller(
        optimizer=optimizer,
        config=cfg,
        method=method,
        total_steps=total_steps,
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
        controller.on_after_backward(loss.item())
        optimizer.step()
        controller.on_batch_end(loss.item())

    controller.on_epoch_end(1.0)
    assert controller.last_lr > 0.0
    assert torch.isfinite(torch.tensor(controller.last_lr))
    return controller


def test_controller_runs_onecycle_modulator() -> None:
    _run_controller_method("ours_onecycle")


def test_random_bounded_modulator_runs() -> None:
    ctrl = _run_controller_method("random_cosine")
    assert abs(ctrl.last_delta) <= 0.1000001


def test_l4_sgd_runs_and_sets_positive_lr() -> None:
    ctrl = _run_controller_method("l4_sgd")
    assert ctrl.stats()["l4_updates"] > 0


def test_hyper_sgd_runs_and_sets_positive_lr() -> None:
    ctrl = _run_controller_method("hyper_sgd")
    assert ctrl.stats()["hyper_updates"] > 0


def test_dadapt_fallback_optimizer_runs() -> None:
    _run_controller_method("dadapt_sgd")


def test_prodigy_fallback_optimizer_runs() -> None:
    _run_controller_method("prodigy")


def test_improved_method_ablations_run() -> None:
    for method in [
        "ours_no_ema_cosine",
        "ours_no_kernel_cosine",
        "ours_no_clip_cosine",
        "ours_deadzone_cosine",
    ]:
        _run_controller_method(method)


def test_shifted_lr_convention_next_step() -> None:
    torch.manual_seed(1)
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        do_finetune=False,
        mod_warmup_steps=0,
        sched_warmup_steps=2,
        use_auto_beta=False,
        beta_fixed=0.5,
    )
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    controller = Controller(optimizer, cfg, "ours_cosine", total_steps=8, steps_per_epoch=4, base_lr=0.1, min_lr=1e-4)
    before = float(optimizer.param_groups[0]["lr"])
    controller.on_batch_end(1.0)
    after_first_loss = float(optimizer.param_groups[0]["lr"])
    assert after_first_loss > 0.0
    assert before != 0.0
