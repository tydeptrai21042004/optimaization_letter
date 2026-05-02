from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer

from .config import ExperimentConfig


def _grad_global_norm(param_groups) -> float:
    total = 0.0
    for group in param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += float(torch.sum(g * g).item())
    return math.sqrt(max(total, 0.0))


class DAdaptSGDFallback(Optimizer):
    """Small self-contained fallback for D-Adaptation-style SGD.

    This is not intended to replace the official D-Adaptation implementation in
    paper experiments.  It provides a stable automatic-LR reference when the
    optional `dadaptation` package is unavailable, so smoke tests and Kaggle runs
    do not crash.  The summary records optimizer_impl='internal_fallback'.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        min_lr: float = 1e-6,
        max_lr: float = 1.0,
        growth_rate: float = 1.02,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            min_lr=min_lr,
            max_lr=max_lr,
            growth_rate=growth_rate,
            eps=eps,
            grad_norm_ema=0.0,
            step_count=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grad_norm = _grad_global_norm([group])
            beta = 0.98
            if group["step_count"] == 0:
                group["grad_norm_ema"] = grad_norm
            else:
                group["grad_norm_ema"] = beta * group["grad_norm_ema"] + (1.0 - beta) * grad_norm

            # Conservative inverse-gradient scale with growth cap.
            target_lr = 1.0 / (group["grad_norm_ema"] + group["eps"])
            prev_lr = float(group["lr"])
            grown_lr = min(prev_lr * group["growth_rate"], target_lr)
            group["lr"] = float(max(group["min_lr"], min(group["max_lr"], grown_lr)))
            group["step_count"] += 1

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                state = self.state[p]
                if momentum != 0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss


class ProdigyFallback(Optimizer):
    """Stable Prodigy-style fallback using AdamW moments plus auto global scale.

    For publishable experiments, install the official `prodigyopt` package.  This
    fallback keeps the code runnable and records optimizer_impl='internal_fallback'.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        min_lr: float = 1e-6,
        max_lr: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            min_lr=min_lr,
            max_lr=max_lr,
            eps=eps,
            step_count=0,
            grad_norm_ema=0.0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            group["step_count"] += 1
            step_count = group["step_count"]
            grad_norm = _grad_global_norm([group])
            if step_count == 1:
                group["grad_norm_ema"] = grad_norm
            else:
                group["grad_norm_ema"] = 0.98 * group["grad_norm_ema"] + 0.02 * grad_norm

            # Auto-scale with a slow warm growth and gradient normalization.
            target_lr = math.sqrt(step_count) / (group["grad_norm_ema"] + group["eps"])
            group["lr"] = float(max(group["min_lr"], min(group["max_lr"], target_lr)))
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - lr * group["weight_decay"])

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step_count
                bias_correction2 = 1 - beta2 ** step_count
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_optimizer_for_method(
    method: str,
    params: Iterable[torch.nn.Parameter],
    base_lr: float,
    config: ExperimentConfig,
) -> Tuple[Optimizer, str]:
    """Build the optimizer for a method and return (optimizer, implementation_tag)."""
    max_lr = float(base_lr * config.max_lr_factor)

    if method == "adamw":
        return (
            torch.optim.AdamW(
                params,
                lr=base_lr,
                weight_decay=config.weight_decay,
            ),
            "torch_adamw",
        )

    if method == "dadapt_sgd":
        try:
            from dadaptation import DAdaptSGD  # type: ignore

            opt = DAdaptSGD(
                params,
                lr=1.0,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
            return opt, "official_dadaptation"
        except Exception:
            opt = DAdaptSGDFallback(
                params,
                lr=base_lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                min_lr=config.min_lr,
                max_lr=max_lr,
                growth_rate=config.dadapt_growth_rate,
            )
            return opt, "internal_fallback"

    if method == "prodigy":
        try:
            from prodigyopt import Prodigy  # type: ignore

            opt = Prodigy(
                params,
                lr=1.0,
                betas=(config.prodigy_beta1, config.prodigy_beta2),
                weight_decay=config.weight_decay,
            )
            return opt, "official_prodigyopt"
        except Exception:
            opt = ProdigyFallback(
                params,
                lr=base_lr,
                betas=(config.prodigy_beta1, config.prodigy_beta2),
                weight_decay=config.weight_decay,
                min_lr=config.min_lr,
                max_lr=max_lr,
            )
            return opt, "internal_fallback"

    # Default optimizer for schedulers, random modulation, L4, and HyperSGD.
    return (
        torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        ),
        "torch_sgd",
    )
