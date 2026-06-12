from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
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
    # Base scheduler warmup.  Backward-compatible with earlier warmup_steps name.
    return _cfg_int(config, "sched_warmup_steps", "warmup_steps", "mod_warmup_steps", default=0)


def _mod_warmup_steps(config: ExperimentConfig) -> int:
    # EMA modulator warmup.  Backward-compatible with earlier warmup_steps name.
    return _cfg_int(config, "mod_warmup_steps", "warmup_steps", "sched_warmup_steps", default=0)


def _grad_norm_sq(optimizer: Optimizer) -> float:
    total = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += float(torch.sum(g * g).item())
    return total


def _flatten_grad_list(optimizer: Optimizer) -> List[torch.Tensor]:
    grads: List[torch.Tensor] = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                grads.append(torch.zeros_like(p.detach()).flatten().cpu())
            else:
                grads.append(p.grad.detach().flatten().cpu().clone())
    return grads


def _grad_dot(current: List[torch.Tensor], previous: List[torch.Tensor]) -> float:
    if not previous or len(current) != len(previous):
        return 0.0
    dot = 0.0
    for g, h in zip(current, previous):
        dot += float(torch.dot(g, h).item())
    return dot


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
    """Delayed weighted h-Hartley--cosine loss-feedback LR modulator.

    The class name is kept for backward compatibility with older scripts, but the
    implemented proposed method is no longer an EMA-only controller.  After batch
    t, the observed loss L_t is mapped to a bounded signal u_t, filtered by a
    delayed weighted h-Hartley--cosine convolution, converted into a backward
    filtered-loss trend, normalized by local loss-signal noise, optionally gated by a trend
    confidence threshold, and clipped before modulating the next base LR:

        eta_{t+1} = r_{t+1} (1 + delta_t).

    The delay D=M+1 is essential.  Direct evaluation of the two-sided generalized
    convolution at index t would require future losses up to t+M+1.  Evaluating it
    at t-D makes the largest required index equal to t, so the next-step LR rule
    remains adapted to the training history.
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
        variant: str = "full",
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.variant = variant
        self.base = BatchBaseSchedule(
            optimizer=optimizer,
            config=config,
            mode=base_mode,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            base_lr=base_lr,
            min_lr=min_lr,
        )

        self.m_win = max(1, _cfg_int(config, "m_win", default=3))
        self.h_step = max(1e-12, _cfg_float(config, "hc_h", "h_step", default=1.0))
        delay_cfg = _cfg_int(config, "hc_delay", default=0)
        self.hc_delay = delay_cfg if delay_cfg > 0 else self.m_win + 1
        # z_t requires nonnegative indices down to t-D-M-1.  q_t=z_{t-1}-z_t
        # therefore needs t >= D+M+2.  With D=M+1 this is 2M+3.
        self.hc_min_warmup = self.hc_delay + self.m_win + 2
        self.mod_warmup_steps = max(0, _mod_warmup_steps(config), self.hc_min_warmup)

        self.hc_kernel = self._build_hc_kernel()
        self.hc_kernel_l1 = float(self.h_step * sum(abs(v) for v in self.hc_kernel.values()))

        self.u_hist: List[float] = []
        self.loss_signal_var: Optional[float] = None
        self.raw_abs_ema: Optional[float] = None
        self.raw_abs_alpha = 0.98
        self.batch_idx = 0

        self.clip_count = 0
        self.emergency_clip_count = 0
        self.total_mod_steps = 0
        self.active_mod_steps = 0
        self.delta_abs_sum = 0.0
        self.beta_eff_sum = 0.0

        self.last_raw = 0.0
        self.last_score = 0.0
        self.last_delta = 0.0
        self.last_base_lr = float(self.base.current_lr)
        self.last_mod_lr = float(self.base.current_lr)
        self.last_beta_eff = 0.0
        self.last_u = 0.0
        self.last_ema = 0.0  # public logging alias: stores filtered HC value z_t.
        self.last_loss_var = 0.0
        self.last_hc_z = 0.0
        self.last_clipped = False
        self.last_emergency_clipped = False
        self.last_gate_active = False

    def _build_hc_kernel(self) -> Dict[int, float]:
        """Build a finite-support symmetric weighted HC kernel.

        The raw weights are rho**abs(m), m=-M,...,M.  They are normalized so that
        a constant input approximately remains constant under the four-shift HC
        convolution: (h/2)*sum_m kappa_m*(4u) = u.
        """
        rho = float(np.clip(_cfg_float(self.config, "rho", default=0.8), 0.0, 0.999999))
        raw = {m: rho ** abs(m) for m in range(-self.m_win, self.m_win + 1)}
        total = sum(raw.values()) + 1e-12
        norm = 1.0 / (2.0 * self.h_step * total)
        return {m: float(w * norm) for m, w in raw.items()}

    def phi(self, z: float) -> float:
        """Bounded monotone loss map used before convolution."""
        z = max(float(z), 0.0)
        if self.variant in {"no_phi", "no_ema"} or not bool(_cfg_get(self.config, "use_phi", default=True)):
            return z
        c_phi = _cfg_float(self.config, "c_phi", default=1.0)
        return z / (c_phi + z + 1e-12)

    def _set_lr(self, lr: float) -> None:
        lr = float(lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _use_hc_convolution(self) -> bool:
        return bool(_cfg_get(self.config, "use_kernel", "use_hc_convolution", default=True)) and self.variant not in {
            "no_kernel",
            "no_hc",
        }

    def _use_noise_normalization(self) -> bool:
        return bool(_cfg_get(self.config, "variance_normalize", "noise_normalize", default=True)) and self.variant not in {
            "no_noise_norm",
        }

    def _use_clipping(self) -> bool:
        return bool(_cfg_get(self.config, "use_clipping", default=True)) and self.variant != "no_clip"

    def _trend_conf_tau(self) -> float:
        # Backward-compatible with the earlier dead_zone_tau argument.
        tau = _cfg_float(self.config, "trend_conf_tau", "dead_zone_tau", default=0.25)
        if self.variant in {"no_gate", "no_deadzone"}:
            return 0.0
        if self.variant == "deadzone" and tau <= 0.0:
            return 0.25
        return max(0.0, tau)

    def _update_signal_variance(self, u: float) -> float:
        var_alpha = _cfg_float(self.config, "var_alpha", default=0.95)
        if len(self.u_hist) < 2:
            residual_sq = 0.0
        else:
            residual_sq = float((self.u_hist[-1] - self.u_hist[-2]) ** 2)
        if self.loss_signal_var is None:
            self.loss_signal_var = residual_sq
        else:
            self.loss_signal_var = var_alpha * self.loss_signal_var + (1.0 - var_alpha) * residual_sq
        self.last_loss_var = float(self.loss_signal_var)
        return float(self.loss_signal_var)

    def _hc_conv_at(self, n: int) -> Optional[float]:
        """Evaluate (kappa *_gamma u)(nh) with finite history.

        Returns None if the delayed index still needs unavailable samples.  With
        the enforced warm-up and D=M+1, this should not happen during active use.
        """
        if n < 0:
            return None
        acc = 0.0
        last = len(self.u_hist) - 1
        for m, kappa_m in self.hc_kernel.items():
            idxs = (n - m - 1, n - m + 1, n + m + 1, n + m - 1)
            if min(idxs) < 0 or max(idxs) > last:
                return None
            acc += kappa_m * sum(self.u_hist[j] for j in idxs)
        return float(0.5 * self.h_step * acc)

    def max_index_used_by_delayed_hc(self, t: int) -> int:
        """Largest signal index used by z_t; useful for causality tests."""
        n = int(t) - self.hc_delay
        return n + self.m_win + 1

    def min_index_used_by_delayed_hc(self, t: int) -> int:
        """Smallest signal index used by z_t; useful for warm-up tests."""
        n = int(t) - self.hc_delay
        return n - self.m_win - 1

    def _hc_trend(self, t: int) -> Optional[float]:
        z_now = self._hc_conv_at(t - self.hc_delay)
        z_prev = self._hc_conv_at(t - 1 - self.hc_delay)
        if z_now is None or z_prev is None:
            return None
        self.last_hc_z = float(z_now)
        self.last_ema = float(z_now)
        q = float(z_prev - z_now)
        if bool(_cfg_get(self.config, "relative_trend", default=False)):
            eps = _cfg_float(self.config, "eps_trend", default=1e-8)
            q = q / (abs(z_prev) + eps)
        return q

    def _simple_causal_trend(self) -> Optional[float]:
        if len(self.u_hist) < 2:
            return None
        q = self.u_hist[-2] - self.u_hist[-1]
        if bool(_cfg_get(self.config, "relative_trend", default=False)):
            eps = _cfg_float(self.config, "eps_trend", default=1e-8)
            q = q / (abs(self.u_hist[-2]) + eps)
        self.last_hc_z = float(self.u_hist[-1])
        self.last_ema = float(self.u_hist[-1])
        return float(q)

    def _beta_eff(self) -> float:
        use_auto_beta = bool(_cfg_get(self.config, "use_auto_beta", default=True))
        beta_fixed = _cfg_float(self.config, "beta_fixed", default=0.08)
        beta_cap = _cfg_float(self.config, "beta_cap", default=3.0)
        target_mean_abs_delta = _cfg_float(self.config, "target_mean_abs_delta", default=0.02)

        if not use_auto_beta:
            return beta_fixed

        if self.raw_abs_ema is None or self.raw_abs_ema < 1e-8:
            return float(min(beta_cap, max(beta_fixed, 0.08)))

        beta = target_mean_abs_delta / (self.raw_abs_ema + 1e-12)
        return float(np.clip(beta, 0.0, beta_cap))

    def on_batch_end(self, loss_value: float) -> None:
        loss_value = float(loss_value)
        gamma = _cfg_float(self.config, "gamma", default=0.1)

        u = self.phi(loss_value)
        self.last_u = float(u)
        self.u_hist.append(float(u))
        var = self._update_signal_variance(u)

        # Advance the base schedule first.  The modulation computed from loss_t is
        # applied to r_{t+1}, so the next optimizer step uses eta_{t+1}.
        self.last_base_lr = self.base.on_batch_end()

        raw_t = 0.0
        score_t = 0.0
        delta_t = 0.0
        clipped = False
        emergency_clipped = False
        beta_eff = 0.0
        gate_active = False

        if self.batch_idx >= self.mod_warmup_steps:
            trend = self._hc_trend(self.batch_idx) if self._use_hc_convolution() else self._simple_causal_trend()
            if trend is not None:
                raw_t = float(trend)
                score_t = raw_t
                if self._use_noise_normalization():
                    eps = _cfg_float(self.config, "eps_trend", default=1e-8)
                    score_t = raw_t / (math.sqrt(max(var, 0.0)) + eps)

                abs_score = abs(score_t)
                if self.raw_abs_ema is None:
                    self.raw_abs_ema = abs_score
                else:
                    self.raw_abs_ema = self.raw_abs_alpha * self.raw_abs_ema + (1.0 - self.raw_abs_alpha) * abs_score

                beta_eff = self._beta_eff()
                tau = self._trend_conf_tau()
                gate_active = abs_score >= tau
                if gate_active:
                    delta_t = beta_eff * score_t

                    if self._use_clipping():
                        if delta_t > gamma:
                            delta_t = gamma
                            clipped = True
                        elif delta_t < -gamma:
                            delta_t = -gamma
                            clipped = True
                    else:
                        # No-clipping ablation still keeps LR positive to avoid invalid optimizer state.
                        floor = _cfg_float(self.config, "emergency_delta_floor", default=-0.95)
                        ceil = _cfg_float(self.config, "emergency_delta_ceiling", default=5.0)
                        if delta_t < floor:
                            delta_t = floor
                            emergency_clipped = True
                        elif delta_t > ceil:
                            delta_t = ceil
                            emergency_clipped = True

                self.total_mod_steps += 1
                self.active_mod_steps += int(gate_active)
                self.clip_count += int(clipped)
                self.emergency_clip_count += int(emergency_clipped)
                self.delta_abs_sum += abs(delta_t)
                self.beta_eff_sum += beta_eff

        self.last_raw = float(raw_t)
        self.last_score = float(score_t)
        self.last_delta = float(delta_t)
        self.last_beta_eff = float(beta_eff)
        self.last_clipped = bool(clipped)
        self.last_emergency_clipped = bool(emergency_clipped)
        self.last_gate_active = bool(gate_active)
        self.last_mod_lr = float(self.last_base_lr * (1.0 + delta_t))
        self._set_lr(self.last_mod_lr)

        self.batch_idx += 1

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        self.last_base_lr = self.base.on_epoch_end(metric)
        self.last_mod_lr = float(self.last_base_lr * (1.0 + self.last_delta))
        self._set_lr(self.last_mod_lr)

    def stats(self) -> Dict[str, float]:
        clip_rate = 0.0 if self.total_mod_steps == 0 else self.clip_count / self.total_mod_steps
        emergency_clip_rate = 0.0 if self.total_mod_steps == 0 else self.emergency_clip_count / self.total_mod_steps
        active_rate = 0.0 if self.total_mod_steps == 0 else self.active_mod_steps / self.total_mod_steps
        delta_mean_abs = 0.0 if self.total_mod_steps == 0 else self.delta_abs_sum / self.total_mod_steps
        beta_eff_mean = 0.0 if self.total_mod_steps == 0 else self.beta_eff_sum / self.total_mod_steps

        return {
            "clip_rate": float(clip_rate),
            "emergency_clip_rate": float(emergency_clip_rate),
            "active_mod_rate": float(active_rate),
            "delta_mean_abs_final": float(delta_mean_abs),
            "beta_eff_mean": float(beta_eff_mean),
            "raw_abs_ema_final": float(self.raw_abs_ema) if self.raw_abs_ema is not None else 0.0,
            "last_hc_z": float(self.last_hc_z),
            "last_score": float(self.last_score),
            "last_u": float(self.last_u),
            "last_loss_var": float(self.last_loss_var),
            "hc_delay": float(self.hc_delay),
            "hc_warmup_steps": float(self.mod_warmup_steps),
            "hc_kernel_l1": float(self.hc_kernel_l1),
            "clip_count": float(self.clip_count),
            "emergency_clip_count": float(self.emergency_clip_count),
            "active_mod_steps": float(self.active_mod_steps),
            "total_mod_steps": float(self.total_mod_steps),
        }


class RandomBoundedModulator:
    """Random clipped perturbation rival.

    eta_{t+1}=r_{t+1}(1+epsilon_t), epsilon_t~Uniform[-gamma,gamma].
    This baseline tests whether improvements are due to the EMA feedback signal
    rather than generic bounded LR perturbation.
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
        self.base = BatchBaseSchedule(optimizer, config, base_mode, total_steps, steps_per_epoch, base_lr, min_lr)
        self.last_lr = float(self.base.current_lr)
        self.last_delta = 0.0
        self.last_raw = 0.0
        self.last_base_lr = float(self.base.current_lr)
        self.last_beta_eff = 0.0
        self.last_ema = 0.0
        self.last_u = 0.0
        self.last_clipped = False
        self.last_emergency_clipped = False
        self.delta_abs_sum = 0.0
        self.total_steps_seen = 0

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(lr)

    def on_batch_end(self, loss_value: float) -> None:
        base_lr = self.base.on_batch_end()
        gamma = _cfg_float(self.config, "random_delta_gamma", default=0.10)
        delta = float(np.random.uniform(-gamma, gamma))
        self.last_base_lr = float(base_lr)
        self.last_delta = delta
        self.last_raw = delta
        self.last_clipped = False
        self.last_emergency_clipped = False
        self.last_lr = float(base_lr * (1.0 + delta))
        self.delta_abs_sum += abs(delta)
        self.total_steps_seen += 1
        self._set_lr(self.last_lr)

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        base_lr = self.base.on_epoch_end(metric)
        self.last_base_lr = float(base_lr)
        self.last_lr = float(base_lr * (1.0 + self.last_delta))
        self._set_lr(self.last_lr)

    def stats(self) -> Dict[str, float]:
        return {
            "random_delta_mean_abs": 0.0 if self.total_steps_seen == 0 else self.delta_abs_sum / self.total_steps_seen,
            "clip_rate": 0.0,
            "delta_mean_abs_final": 0.0 if self.total_steps_seen == 0 else self.delta_abs_sum / self.total_steps_seen,
        }


class L4StepController:
    """Practical L4-style loss-based step-size controller.

    This self-contained implementation computes the LR after backward and before
    optimizer.step(): lr = alpha * max(loss - gamma*loss_floor, 0) / ||g||^2.
    It is clipped for numerical safety.  For publication-grade comparison, you may
    replace this by the authors' official implementation if desired.
    """

    def __init__(self, optimizer: Optimizer, config: ExperimentConfig, base_lr: float, min_lr: float) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.max_lr = float(base_lr * _cfg_float(config, "max_lr_factor", default=10.0))
        self.loss_floor: Optional[float] = None
        self.last_lr = float(optimizer.param_groups[0]["lr"])
        self.last_delta = 0.0
        self.last_raw = 0.0
        self.update_count = 0

    def _set_lr(self, lr: float) -> None:
        lr = float(np.clip(lr, self.min_lr, self.max_lr))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.last_lr = lr

    def on_after_backward(self, loss_value: float) -> None:
        loss_value = float(loss_value)
        if self.loss_floor is None:
            self.loss_floor = loss_value
        else:
            self.loss_floor = min(self.loss_floor, loss_value)

        alpha = _cfg_float(self.config, "l4_alpha", default=0.15)
        gamma = _cfg_float(self.config, "l4_gamma", default=0.90)
        eps = _cfg_float(self.config, "l4_eps", default=1e-12)
        g2 = max(_grad_norm_sq(self.optimizer), eps)
        target = gamma * float(self.loss_floor)
        raw_gap = max(loss_value - target, 0.0)
        lr = alpha * raw_gap / g2

        self.last_raw = raw_gap
        self._set_lr(lr)
        self.last_delta = self.last_lr / self.base_lr - 1.0 if self.base_lr > 0 else 0.0
        self.update_count += 1

    def on_batch_end(self, loss_value: float) -> None:
        # LR was already set before optimizer.step().
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])

    def stats(self) -> Dict[str, float]:
        return {
            "l4_loss_floor": 0.0 if self.loss_floor is None else float(self.loss_floor),
            "l4_updates": float(self.update_count),
        }


class HyperGradientController:
    """Scalar hypergradient LR adaptation for SGD.

    Uses lr_t = clip(lr_{t-1} + hyper_lr * <g_t, g_{t-1}>).  The update is
    applied after backward and before optimizer.step(), so the current batch uses
    the hypergradient-updated LR.
    """

    def __init__(self, optimizer: Optimizer, config: ExperimentConfig, base_lr: float, min_lr: float) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_lr = float(base_lr)
        self.min_lr = max(float(min_lr), base_lr * _cfg_float(config, "hyper_min_lr_factor", default=0.01))
        self.max_lr = base_lr * _cfg_float(config, "hyper_max_lr_factor", default=10.0)
        self.prev_grads: List[torch.Tensor] = []
        self.curr_grads: List[torch.Tensor] = []
        self.last_lr = float(optimizer.param_groups[0]["lr"])
        self.last_delta = 0.0
        self.last_raw = 0.0
        self.update_count = 0

    def _set_lr(self, lr: float) -> None:
        lr = float(np.clip(lr, self.min_lr, self.max_lr))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.last_lr = lr

    def on_after_backward(self, loss_value: float) -> None:
        self.curr_grads = _flatten_grad_list(self.optimizer)
        dot = _grad_dot(self.curr_grads, self.prev_grads)
        hyper_lr = _cfg_float(self.config, "hyper_lr", default=1e-6)
        new_lr = float(self.optimizer.param_groups[0]["lr"]) + hyper_lr * dot
        self.last_raw = float(dot)
        self._set_lr(new_lr)
        self.last_delta = self.last_lr / self.base_lr - 1.0 if self.base_lr > 0 else 0.0
        self.update_count += 1

    def on_batch_end(self, loss_value: float) -> None:
        self.prev_grads = [g.clone() for g in self.curr_grads]
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        self.last_lr = float(self.optimizer.param_groups[0]["lr"])

    def stats(self) -> Dict[str, float]:
        return {"hyper_updates": float(self.update_count)}


class OptimizerOnlyController:
    """No scheduler: optimizer internally controls LR/adaptation."""

    def __init__(self, optimizer: Optimizer, method: str) -> None:
        self.optimizer = optimizer
        self.method = method
        self.last_lr = float(optimizer.param_groups[0].get("lr", 0.0))
        self.last_delta = 0.0
        self.last_raw = 0.0

    def on_after_backward(self, loss_value: float) -> None:
        pass

    def on_batch_end(self, loss_value: float) -> None:
        self.last_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        self.last_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

    def stats(self) -> Dict[str, float]:
        return {}


class Controller:
    """Unified wrapper around base schedulers, proposed modulator, and rival optimizers.

    Public attributes are intentionally normalized so the training loop can log the same
    batch-level diagnostics for every method:
      last_base_lr, last_lr, last_delta, last_raw, last_beta_eff, last_ema_loss,
      last_u_signal, last_clipped, last_emergency_clipped, and last_grad_norm_sq.
    """

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
        self.optimizer = optimizer
        self.config = config
        self.method = method
        self.last_lr = float(optimizer.param_groups[0].get("lr", 0.0))
        self.last_delta = 0.0
        self.last_raw = 0.0
        self.last_base_lr = float(self.last_lr)
        self.last_beta_eff = 0.0
        self.last_ema_loss = 0.0
        self.last_u_signal = 0.0
        self.last_clipped = False
        self.last_emergency_clipped = False
        self.last_grad_norm_sq = 0.0

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
            "ours_plateau",
            "ours_no_hc_cosine",
            "ours_no_noise_norm_cosine",
            "ours_no_gate_cosine",
            "ours_no_gate_plateau",
            "ours_no_clip_plateau",
            "ours_no_noise_norm_plateau",
            "ours_no_hc_plateau",
            "ours_with_gate_plateau",
            "ours_no_phi_cosine",
            "ours_no_clip_cosine",
            # Backward-compatible aliases from the old EMA manuscript version.
            "ours_no_ema_cosine",
            "ours_no_kernel_cosine",
            "ours_deadzone_cosine",
        }:
            self.kind = "mod"
            base_mode = {
                "ours_cosine": "cosine",
                "ours_onecycle": "onecycle",
                "ours_warmup_cosine": "warmup_cosine",
                "ours_plateau": "plateau",
                "ours_no_hc_cosine": "cosine",
                "ours_no_noise_norm_cosine": "cosine",
                "ours_no_gate_cosine": "cosine",
                "ours_no_gate_plateau": "plateau",
                "ours_no_clip_plateau": "plateau",
                "ours_no_noise_norm_plateau": "plateau",
                "ours_no_hc_plateau": "plateau",
                "ours_with_gate_plateau": "plateau",
                "ours_no_phi_cosine": "cosine",
                "ours_no_clip_cosine": "cosine",
                "ours_no_ema_cosine": "cosine",
                "ours_no_kernel_cosine": "cosine",
                "ours_deadzone_cosine": "cosine",
            }[method]
            variant = {
                "ours_no_hc_cosine": "no_hc",
                "ours_no_noise_norm_cosine": "no_noise_norm",
                "ours_no_gate_cosine": "no_gate",
                "ours_plateau": "no_gate",
                "ours_no_gate_plateau": "no_gate",
                "ours_no_clip_plateau": "no_clip",
                "ours_no_noise_norm_plateau": "no_noise_norm",
                "ours_no_hc_plateau": "no_hc",
                "ours_with_gate_plateau": "full",
                "ours_no_phi_cosine": "no_phi",
                "ours_no_clip_cosine": "no_clip",
                # Old aliases.
                "ours_no_ema_cosine": "no_phi",
                "ours_no_kernel_cosine": "no_hc",
                "ours_deadzone_cosine": "deadzone",
            }.get(method, "full")

            self.mod = EMALossModulator(
                optimizer=optimizer,
                config=config,
                base_mode=base_mode,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                base_lr=base_lr,
                min_lr=min_lr,
                variant=variant,
            )
            self.last_lr = self.mod.last_mod_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        elif method in {"random_cosine", "random_onecycle", "random_warmup_cosine", "random_plateau"}:
            self.kind = "random"
            base_mode = {
                "random_cosine": "cosine",
                "random_onecycle": "onecycle",
                "random_warmup_cosine": "warmup_cosine",
                "random_plateau": "plateau",
            }[method]
            self.random = RandomBoundedModulator(
                optimizer=optimizer,
                config=config,
                base_mode=base_mode,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                base_lr=base_lr,
                min_lr=min_lr,
            )
            self.last_lr = self.random.last_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        elif method == "l4_sgd":
            self.kind = "l4"
            self.l4 = L4StepController(optimizer, config, base_lr=base_lr, min_lr=min_lr)
            self.last_lr = self.l4.last_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        elif method == "hyper_sgd":
            self.kind = "hyper"
            self.hyper = HyperGradientController(optimizer, config, base_lr=base_lr, min_lr=min_lr)
            self.last_lr = self.hyper.last_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        elif method in {"dadapt_sgd", "prodigy", "adamw"}:
            self.kind = "optimizer_only"
            self.opt_only = OptimizerOnlyController(optimizer, method)
            self.last_lr = self.opt_only.last_lr
            self.last_delta = 0.0
            self.last_raw = 0.0

        else:
            raise ValueError(f"Unknown method: {method}")

        self._sync_public_state()

    def _sync_public_state(self) -> None:
        """Synchronize public logging attributes after each controller action."""
        if self.kind == "base":
            self.last_base_lr = float(self.base.current_lr)
            self.last_beta_eff = 0.0
            self.last_ema_loss = 0.0
            self.last_u_signal = 0.0
            self.last_clipped = False
            self.last_emergency_clipped = False
        elif self.kind == "mod":
            self.last_base_lr = float(self.mod.last_base_lr)
            self.last_beta_eff = float(self.mod.last_beta_eff)
            self.last_ema_loss = float(self.mod.last_ema)
            self.last_u_signal = float(self.mod.last_u)
            self.last_clipped = bool(self.mod.last_clipped)
            self.last_emergency_clipped = bool(self.mod.last_emergency_clipped)
        elif self.kind == "random":
            self.last_base_lr = float(self.random.last_base_lr)
            self.last_beta_eff = 0.0
            self.last_ema_loss = 0.0
            self.last_u_signal = 0.0
            self.last_clipped = False
            self.last_emergency_clipped = False
        else:
            self.last_base_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self.last_beta_eff = 0.0
            self.last_ema_loss = 0.0
            self.last_u_signal = 0.0
            self.last_clipped = False
            self.last_emergency_clipped = False

    def set_grad_norm_sq(self, value: float) -> None:
        self.last_grad_norm_sq = float(value)

    def on_after_backward(self, loss_value: float) -> None:
        if self.kind == "l4":
            self.l4.on_after_backward(loss_value)
            self.last_lr = self.l4.last_lr
            self.last_delta = self.l4.last_delta
            self.last_raw = self.l4.last_raw
        elif self.kind == "hyper":
            self.hyper.on_after_backward(loss_value)
            self.last_lr = self.hyper.last_lr
            self.last_delta = self.hyper.last_delta
            self.last_raw = self.hyper.last_raw
        elif self.kind == "optimizer_only":
            self.opt_only.on_after_backward(loss_value)
        self._sync_public_state()

    def on_batch_end(self, loss_value: float) -> None:
        if self.kind == "base":
            self.last_lr = self.base.on_batch_end()
            self.last_delta = 0.0
            self.last_raw = 0.0
        elif self.kind == "mod":
            self.mod.on_batch_end(loss_value)
            self.last_lr = self.mod.last_mod_lr
            self.last_delta = self.mod.last_delta
            self.last_raw = self.mod.last_raw
        elif self.kind == "random":
            self.random.on_batch_end(loss_value)
            self.last_lr = self.random.last_lr
            self.last_delta = self.random.last_delta
            self.last_raw = self.random.last_raw
        elif self.kind == "l4":
            self.l4.on_batch_end(loss_value)
            self.last_lr = self.l4.last_lr
            self.last_delta = self.l4.last_delta
            self.last_raw = self.l4.last_raw
        elif self.kind == "hyper":
            self.hyper.on_batch_end(loss_value)
            self.last_lr = self.hyper.last_lr
            self.last_delta = self.hyper.last_delta
            self.last_raw = self.hyper.last_raw
        elif self.kind == "optimizer_only":
            self.opt_only.on_batch_end(loss_value)
            self.last_lr = self.opt_only.last_lr
            self.last_delta = 0.0
            self.last_raw = 0.0
        self._sync_public_state()

    def on_epoch_end(self, metric: Optional[float] = None) -> None:
        if self.kind == "base":
            self.last_lr = self.base.on_epoch_end(metric)
        elif self.kind == "mod":
            self.mod.on_epoch_end(metric)
            self.last_lr = self.mod.last_mod_lr
            self.last_delta = self.mod.last_delta
            self.last_raw = self.mod.last_raw
        elif self.kind == "random":
            self.random.on_epoch_end(metric)
            self.last_lr = self.random.last_lr
            self.last_delta = self.random.last_delta
            self.last_raw = self.random.last_raw
        elif self.kind == "l4":
            self.l4.on_epoch_end(metric)
            self.last_lr = self.l4.last_lr
            self.last_delta = self.l4.last_delta
            self.last_raw = self.l4.last_raw
        elif self.kind == "hyper":
            self.hyper.on_epoch_end(metric)
            self.last_lr = self.hyper.last_lr
            self.last_delta = self.hyper.last_delta
            self.last_raw = self.hyper.last_raw
        elif self.kind == "optimizer_only":
            self.opt_only.on_epoch_end(metric)
            self.last_lr = self.opt_only.last_lr
        self._sync_public_state()

    def stats(self) -> Dict[str, float]:
        if self.kind == "base":
            return {}
        if self.kind == "mod":
            return self.mod.stats()
        if self.kind == "random":
            return self.random.stats()
        if self.kind == "l4":
            return self.l4.stats()
        if self.kind == "hyper":
            return self.hyper.stats()
        if self.kind == "optimizer_only":
            return self.opt_only.stats()
        return {}
