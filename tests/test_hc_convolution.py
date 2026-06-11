from __future__ import annotations

import math

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import EMALossModulator


def _make_modulator(
    *,
    m_win: int = 3,
    gamma: float = 0.10,
    trend_conf_tau: float = 0.0,
    variance_normalize: bool = True,
    use_auto_beta: bool = False,
    beta_fixed: float = 0.5,
    mod_warmup_steps: int = 0,
    use_clipping: bool = True,
    variant: str = "full",
) -> EMALossModulator:
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        m_win=m_win,
        gamma=gamma,
        trend_conf_tau=trend_conf_tau,
        variance_normalize=variance_normalize,
        use_auto_beta=use_auto_beta,
        beta_fixed=beta_fixed,
        mod_warmup_steps=mod_warmup_steps,
        sched_warmup_steps=1,
        use_clipping=use_clipping,
    )
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=0.1)
    return EMALossModulator(
        optimizer=optimizer,
        config=cfg,
        base_mode="cosine",
        total_steps=80,
        steps_per_epoch=20,
        base_lr=0.1,
        min_lr=1e-4,
        variant=variant,
    )


def test_delayed_hc_indices_are_adapted_for_many_windows() -> None:
    for m_win in [1, 2, 3, 5, 8]:
        mod = _make_modulator(m_win=m_win)
        first_active_t = int(mod.mod_warmup_steps)
        for t in range(first_active_t, first_active_t + 10):
            assert mod.max_index_used_by_delayed_hc(t) <= t
            assert mod.min_index_used_by_delayed_hc(t) >= 0


def test_direct_hc_evaluation_would_use_future_losses() -> None:
    m_win = 4
    t = 20
    direct_max_index = t + m_win + 1
    delayed_max_index = t - (m_win + 1) + m_win + 1
    assert direct_max_index > t
    assert delayed_max_index == t


def test_hc_convolution_preserves_constant_signal_after_warmup() -> None:
    mod = _make_modulator(m_win=3)
    constant_value = 0.75
    mod.u_hist = [constant_value] * 40
    n = 20
    z = mod._hc_conv_at(n)
    assert z is not None
    assert math.isclose(float(z), constant_value, rel_tol=1e-6, abs_tol=1e-6)


def test_hc_convolution_returns_none_when_history_is_incomplete() -> None:
    mod = _make_modulator(m_win=3)
    mod.u_hist = [0.1, 0.2, 0.3]
    assert mod._hc_conv_at(2) is None


def test_clipped_modulation_never_exceeds_gamma() -> None:
    gamma = 0.05
    mod = _make_modulator(
        m_win=2,
        gamma=gamma,
        trend_conf_tau=0.0,
        variance_normalize=False,
        use_auto_beta=False,
        beta_fixed=100.0,
        mod_warmup_steps=0,
    )
    losses = [3.0] * 10 + [2.5, 2.0, 1.4, 1.0, 0.7, 0.5, 0.4, 0.3]
    for loss in losses:
        mod.on_batch_end(loss)
    assert abs(mod.last_delta) <= gamma + 1e-12
    assert mod.last_mod_lr > 0.0


def test_confidence_gate_suppresses_flat_loss_signal() -> None:
    mod = _make_modulator(
        m_win=2,
        trend_conf_tau=0.25,
        variance_normalize=True,
        use_auto_beta=False,
        beta_fixed=1.0,
        mod_warmup_steps=0,
    )
    for _ in range(30):
        mod.on_batch_end(1.0)
    assert mod.total_mod_steps > 0
    assert mod.active_mod_steps == 0
    assert abs(mod.last_delta) == 0.0


def test_no_hc_ablation_uses_simple_causal_trend_not_future_history() -> None:
    mod = _make_modulator(
        m_win=3,
        trend_conf_tau=0.0,
        variance_normalize=False,
        use_auto_beta=False,
        beta_fixed=1.0,
        variant="no_hc",
    )
    mod.u_hist = [0.8, 0.7]
    trend = mod._simple_causal_trend()
    assert trend is not None
    assert trend > 0.0
