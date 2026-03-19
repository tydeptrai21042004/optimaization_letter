from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
)

from .config import ExperimentConfig


def _cfg_get(config: ExperimentConfig, *names: str, default=None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _cfg_int(config: ExperimentConfig, *names: str, default: int = 0) -> int:
    return int(_cfg_get(config, *names, default=default))


def _cfg_float(config: ExperimentConfig, *names: str, default: float = 0.0) -> float:
    return float(_cfg_get(config, *names, default=default))


def _cfg_str(config: ExperimentConfig, *names: str, default: str = "") -> str:
    return str(_cfg_get(config, *names, default=default))


def _sched_warmup_steps(config: ExperimentConfig) -> int:
    # Base scheduler warmup
    return _cfg_int(config, "sched_warmup_steps", "warmup_steps", "mod_warmup_steps", default=0)


def _mod_warmup_steps(config: ExperimentConfig) -> int:
    # EMA modulator warmup
    return _cfg_int(config, "mod_warmup_steps", "warmup_steps", "sched_warmup_steps", default=0)


class BatchBaseSchedule:
    def __init__(
        self,
        optimizer: Optimizer,
        config: ExperimentConfig,
        mode: str,
        total_steps: int,
        steps_per_epoch: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.mode = mode
        self.total_steps = max(1, int(total_steps))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.step_idx = 0

        if mode == "onecycle":
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=base_lr,
                total_steps=self.total_steps,
                pct_start=0.3,
                anneal_strategy="cos",
            )
            self.current_lr = float(optimizer.param_groups[0]["lr"])

        elif mode == "warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, _cfg_int(config, "restart_t0_steps", "restart_t0", default=10)),
                T_mult=max(1, _cfg_int(config, "restart_t_mult", "t_mult", default=1)),
                eta_min=self.min_lr,
            )
            self.current_lr = float(optimizer.param_groups[0]["lr"])

        elif mode == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=_cfg_str(config, "plateau_mode", default="min"),
                factor=_cfg_float(config, "plateau_factor", default=0.5),
                patience=_cfg_int(config, "plateau_patience", default=2),
                threshold=_cfg_float(config, "plateau_threshold", default=1e-4),
                threshold_mode=_cfg_str(config, "plateau_threshold_mode", default="rel"),
                cooldown=_cfg_int(config, "plateau_cooldown", default=0),
                min_lr=self.min_lr,
            )
            self.current_lr = float(optimizer.param_groups[0]["lr"])

        else:
            self.scheduler = None
            self.current_lr = self.lr_at(0)
            self._set_lr(self.current_lr)

    def _set_lr(self, lr: float) -> None:
        lr = float(lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def lr_at(self, step_idx: int) -> float:
        step_idx = min(max(0, int(step_idx)), self.total_steps)

        if self.mode == "constant":
            return self.base_lr

        if self.mode == "step":
            epoch_idx = step_idx // self.steps_per_epoch
            step_size_epochs = max(1, _cfg_int(self.config, "step_size_epochs", default=30))
            step_gamma = _cfg_float(self.config, "step_gamma", default=0.1)
            decay_count = epoch_idx // step_size_epochs
            return self.base_lr * (step_gamma ** decay_count)

        if self.mode == "cosine":
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * step_idx / self.total_steps)
            )

        if self.mode == "warmup_cosine":
            warmup_steps = min(max(1, _sched_warmup_steps(self.config)), self.total_steps)
            warmup_start_factor = _cfg_float(self.config, "warmup_start_factor", default=0.1)
            warmup_start_lr = self.base_lr * warmup_start_factor

            if step_idx <= warmup_steps:
                p = step_idx / warmup_steps
                return warmup_start_lr + p * (self.base_lr - warmup_start_lr)

            cosine_steps = max(1, self.total_steps - warmup_steps)
            t = min(step_idx - warmup_steps, cosine_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * t / cosine_steps)
            )

        raise ValueError(f"Unsupported base schedule: {self.mode}")

    def on_batch_end(self) -> float:
        self.step_idx += 1

        if self.mode == "onecycle":
            assert self.scheduler is not None
            self.scheduler.step()
            self.current_lr = float(self.optimizer.param_groups[0]["lr"])

        elif self.mode == "warm_restarts":
            assert self.scheduler is not None
            self.scheduler.step()
            self.current_lr = float(self.optimizer.param_groups[0]["lr"])

        elif self.mode == "plateau":
            self.current_lr = float(self.optimizer.param_groups[0]["lr"])

        else:
            self.current_lr = self.lr_at(self.step_idx)
            self._set_lr(self.current_lr)

        return self.current_lr

    def on_epoch_end(self, metric: Optional[float] = None) -> float:
        if self.mode != "plateau":
            return self.current_lr

        if metric is None:
            raise ValueError(
                "plateau scheduler requires a monitored metric at epoch end "
                "(e.g. validation loss)."
            )

        assert self.scheduler is not None
        self.scheduler.step(float(metric))
        self.current_lr = float(self.optimizer.param_groups[0]["lr"])
        return self.current_lr


class EMALossModulator:
    """Bounded EMA-loss-based learning-rate modulator.

    raw_t = sum_m w_m (u_{t-m} - u_t)
    delta_t = clip(beta_eff * raw_t, -gamma, gamma)
    executed_lr = base_lr * (1 + delta_t)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: ExperimentConfig,
        base_mode: str,
        total_steps: int,
        steps_per_epoch: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base = BatchBaseSchedule(
            optimizer=optimizer,
            config=config,
            mode=base_mode,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            base_lr=base_lr,
            min_lr=min_lr,
        )

        self.m_win = max(1, _cfg_int(config, "m_win", default=5))
        self.ema: Optional[float] = None
        self.hist = deque(maxlen=self.m_win + 1)
        self.batch_idx = 0

        rho = _cfg_float(config, "rho", default=0.9)
        z = sum(rho ** (j - 1) for j in range(1, self.m_win + 1))
        self.kernel = [
            (rho ** (m - 1)) / (z + 1e-12)
            for m in range(1, self.m_win + 1)
        ]

        self.raw_abs_ema: Optional[float] = None
        self.raw_abs_alpha = 0.98

        self.clip_count = 0
        self.total_mod_steps = 0
        self.delta_abs_sum = 0.0
        self.beta_eff_sum = 0.0

        self.last_raw = 0.0
        self.last_delta = 0.0
        self.last_base_lr = float(self.base.current_lr)
        self.last_mod_lr = float(self.base.current_lr)
        self.last_beta_eff = 0.0

    def phi(self, z: float) -> float:
        z = max(float(z), 0.0)
        c_phi = _cfg_float(self.config, "c_phi", default=1.0)
        return z / (c_phi + z + 1e-12)

    def _set_lr(self, lr: float) -> None:
        lr = float(lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _raw(self, u_now: float) -> float:
        raw = 0.0
        scale = 0.0
        normalize_raw = bool(_cfg_get(self.config, "normalize_raw", default=False))

        for m in range(1, self.m_win + 1):
            u_prev = self.hist[-1 - m]
            diff = u_prev - u_now
            w = self.kernel[m - 1]
            raw += w * diff
            scale += w * abs(diff)

        return raw / (scale + 1e-12) if normalize_raw else raw

    def _beta_eff(self) -> float:
        use_auto_beta = bool(_cfg_get(self.config, "use_auto_beta", default=False))
        beta_fixed = _cfg_float(self.config, "beta_fixed", default=0.5)
        beta_cap = _cfg_float(self.config, "beta_cap", default=1.0)
        target_mean_abs_delta = _cfg_float(self.config, "target_mean_abs_delta", default=0.05)

        if not use_auto_beta:
            return beta_fixed

        if self.raw_abs_ema is None or self.raw_abs_ema < 1e-8:
            return float(min(beta_cap, max(beta_fixed, 0.5)))

        beta = target_mean_abs_delta / (self.raw_abs_ema + 1e-12)
        return float(np.clip(beta, 0.0, beta_cap))

    def on_batch_end(self, loss_value: float) -> None:
        loss_value = float(loss_value)

        alpha = _cfg_float(self.config, "alpha", default=0.9)
        gamma = _cfg_float(self.config, "gamma", default=0.1)
        mod_warmup = max(0, _mod_warmup_steps(self.config))

        if self.ema is None:
            self.ema = loss_value
        else:
            self.ema = alpha * self.ema + (1.0 - alpha) * loss_value

        u = self.phi(self.ema)
        self.hist.append(u)

        self.last_base_lr = self.base.on_batch_end()

        raw_t = 0.0
        delta_t = 0.0
        clipped = False
        beta_eff = 0.0

        if self.batch_idx >= mod_warmup and len(self.hist) >= self.m_win + 1:
            raw_t = self._raw(u)
            abs_raw = abs(raw_t)

            if self.raw_abs_ema is None:
                self.raw_abs_ema = abs_raw
            else:
                self.raw_abs_ema = (
                    self.raw_abs_alpha * self.raw_abs_ema
                    + (1.0 - self.raw_abs_alpha) * abs_raw
                )

            beta_eff = self._beta_eff()
            delta_t = beta_eff * raw_t

            if delta_t > gamma:
                delta_t = gamma
                clipped = True
            elif delta_t < -gamma:
                delta_t = -gamma
                clipped = True

            self.total_mod_steps += 1
            self.clip_count += int(clipped)
            self.delta_abs_sum += abs(delta_t)
            self.beta_eff_sum += beta_eff

        self.last_raw = float(raw_t)
        self.last_delta = float(delta_t)
        self.last_beta_eff = float(beta_eff)
        self.last_mod_lr = float(self.last_base_lr * (1.0 + delta_t))
        self._set_lr(self.last_mod_lr)

        self.batch_idx += 1

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        self.last_base_lr = self.base.on_epoch_end(metric)
        self.last_mod_lr = float(self.last_base_lr * (1.0 + self.last_delta))
        self._set_lr(self.last_mod_lr)

    def stats(self) -> Dict[str, float]:
        clip_rate = 0.0 if self.total_mod_steps == 0 else self.clip_count / self.total_mod_steps
        delta_mean_abs = 0.0 if self.total_mod_steps == 0 else self.delta_abs_sum / self.total_mod_steps
        beta_eff_mean = 0.0 if self.total_mod_steps == 0 else self.beta_eff_sum / self.total_mod_steps

        return {
            "clip_rate": float(clip_rate),
            "delta_mean_abs_final": float(delta_mean_abs),
            "beta_eff_mean": float(beta_eff_mean),
            "raw_abs_ema_final": float(self.raw_abs_ema) if self.raw_abs_ema is not None else 0.0,
        }


class Controller:
    def __init__(
        self,
        optimizer: Optimizer,
        config: ExperimentConfig,
        method: str,
        total_steps: int,
        steps_per_epoch: int,
        base_lr: float,
        min_lr: float,
    ) -> None:
        self.method = method

        if method in {
            "constant",
            "step",
            "cosine",
            "onecycle",
            "warmup_cosine",
            "warm_restarts",
            "plateau",
        }:
            self.kind = "base"
            self.base = BatchBaseSchedule(
                optimizer=optimizer,
                config=config,
                mode=method,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                base_lr=base_lr,
                min_lr=min_lr,
            )
            self.last_lr = self.base.current_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        elif method in {
            "ours_cosine",
            "ours_onecycle",
            "ours_warmup_cosine",
        }:
            self.kind = "mod"
            base_mode = {
                "ours_cosine": "cosine",
                "ours_onecycle": "onecycle",
                "ours_warmup_cosine": "warmup_cosine",
            }[method]

            self.mod = EMALossModulator(
                optimizer=optimizer,
                config=config,
                base_mode=base_mode,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                base_lr=base_lr,
                min_lr=min_lr,
            )
            self.last_lr = self.mod.last_mod_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        else:
            raise ValueError(f"Unknown method: {method}")

    def on_batch_end(self, loss_value: float) -> None:
        if self.kind == "base":
            self.last_lr = self.base.on_batch_end()
            self.last_delta = 0.0
            self.last_raw = 0.0
        else:
            self.mod.on_batch_end(loss_value)
            self.last_lr = self.mod.last_mod_lr
            self.last_delta = self.mod.last_delta
            self.last_raw = self.mod.last_raw

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        if self.kind == "base":
            self.last_lr = self.base.on_epoch_end(metric)
        else:
            self.mod.on_epoch_end(metric)
            self.last_lr = self.mod.last_mod_lr
            self.last_delta = self.mod.last_delta
            self.last_raw = self.mod.last_raw

    def stats(self) -> Dict[str, float]:
        return {} if self.kind == "base" else self.mod.stats()
