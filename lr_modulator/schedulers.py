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

        # Scheduler-backed modes
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
                T_0=max(1, int(config.restart_t0_steps)),
                T_mult=max(1, int(config.restart_t_mult)),
                eta_min=self.min_lr,
            )
            self.current_lr = float(optimizer.param_groups[0]["lr"])

        elif mode == "plateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config.plateau_mode,
                factor=config.plateau_factor,
                patience=config.plateau_patience,
                threshold=config.plateau_threshold,
                threshold_mode=config.plateau_threshold_mode,
                cooldown=config.plateau_cooldown,
                min_lr=self.min_lr,
            )
            self.current_lr = float(optimizer.param_groups[0]["lr"])

        # Closed-form/manual modes
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
            decay_count = epoch_idx // self.config.step_size_epochs
            return self.base_lr * (self.config.step_gamma ** decay_count)

        if self.mode == "cosine":
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * step_idx / self.total_steps)
            )

        if self.mode == "warmup_cosine":
            warmup_steps = max(1, int(self.config.warmup_steps))
            warmup_start_lr = self.base_lr * float(self.config.warmup_start_factor)

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
            # No batch update here.
            # LR changes only when on_epoch_end(metric) is called.
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

        self.ema: Optional[float] = None
        self.hist = deque(maxlen=config.m_win + 1)
        self.batch_idx = 0

        z = sum(config.rho ** (j - 1) for j in range(1, config.m_win + 1))
        self.kernel = [
            (config.rho ** (m - 1)) / (z + 1e-12)
            for m in range(1, config.m_win + 1)
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
        return z / (self.config.c_phi + z + 1e-12)

    def _set_lr(self, lr: float) -> None:
        lr = float(lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _raw(self, u_now: float) -> float:
        raw = 0.0
        scale = 0.0
        for m in range(1, self.config.m_win + 1):
            u_prev = self.hist[-1 - m]
            diff = u_prev - u_now
            w = self.kernel[m - 1]
            raw += w * diff
            scale += w * abs(diff)
        return raw / (scale + 1e-12) if self.config.normalize_raw else raw

    def _beta_eff(self) -> float:
        if not self.config.use_auto_beta:
            return float(self.config.beta_fixed)

        if self.raw_abs_ema is None or self.raw_abs_ema < 1e-8:
            return float(
                min(self.config.beta_cap, max(self.config.beta_fixed, 0.5))
            )

        beta = self.config.target_mean_abs_delta / (self.raw_abs_ema + 1e-12)
        return float(np.clip(beta, 0.0, self.config.beta_cap))

    def on_batch_end(self, loss_value: float) -> None:
        loss_value = float(loss_value)

        if self.ema is None:
            self.ema = loss_value
        else:
            self.ema = self.config.alpha * self.ema + (1.0 - self.config.alpha) * loss_value

        u = self.phi(self.ema)
        self.hist.append(u)

        self.last_base_lr = self.base.on_batch_end()

        raw_t = 0.0
        delta_t = 0.0
        clipped = False
        beta_eff = 0.0

        if (
            self.batch_idx >= self.config.warmup_steps
            and len(self.hist) >= self.config.m_win + 1
        ):
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

            if delta_t > self.config.gamma:
                delta_t = self.config.gamma
                clipped = True
            elif delta_t < -self.config.gamma:
                delta_t = -self.config.gamma
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
        # Useful if the underlying base schedule is epoch/metric-driven
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
