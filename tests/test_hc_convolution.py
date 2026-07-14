from __future__ import annotations

import math

import torch

from lr_modulator.config import ExperimentConfig
from lr_modulator.schedulers import EMALossModulator


def _make_modulator(
    *,
    m_win: int = 3,
    alpha: float = 0.5,
    gamma: float = 0.10,
    trend_conf_tau: float = 0.0,
    variance_normalize: bool = False,
    use_auto_beta: bool = False,
    beta_fixed: float = 0.02,
    mod_warmup_steps: int = 0,
    use_clipping: bool = True,
    variant: str = "full",
) -> EMALossModulator:
    cfg = ExperimentConfig(
        use_amp=False,
        num_workers=0,
        download=False,
        do_finetune=False,
        alpha=alpha,
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


def test_causal_operator_uses_exactly_t_minus_m_through_t() -> None:
    for m_win in [1, 2, 3, 5, 8]:
        mod = _make_modulator(m_win=m_win)
        for t in range(m_win, m_win + 10):
            first, last = mod.causal_index_range(t)
            assert first == t - m_win
            assert last == t
            assert mod.max_index_used_by_delayed_hc(t) == t
            assert mod.min_index_used_by_delayed_hc(t) == t - m_win


def test_normalized_causal_kernel_has_unit_truncated_mass() -> None:
    mod = _make_modulator(m_win=5)
    mass = 0.5 * mod.h_step * sum(mod.causal_weights.values())
    assert math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mod.causal_kernel_norm, 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_constant_signal_has_zero_feedback() -> None:
    mod = _make_modulator(m_win=3)
    mod.u_hist = [0.75] * 10
    raw = mod._direct_causal_feedback_at(9)
    assert raw is not None
    assert math.isclose(raw, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_decreasing_signal_has_nonnegative_feedback() -> None:
    mod = _make_modulator(m_win=3)
    mod.u_hist = [1.0, 0.9, 0.8, 0.7]
    raw = mod._direct_causal_feedback_at(3)
    assert raw is not None
    assert raw > 0.0


def test_increasing_signal_has_nonpositive_feedback() -> None:
    mod = _make_modulator(m_win=3)
    mod.u_hist = [0.2, 0.3, 0.4, 0.5]
    raw = mod._direct_causal_feedback_at(3)
    assert raw is not None
    assert raw < 0.0


def test_feedback_returns_none_when_history_is_incomplete() -> None:
    mod = _make_modulator(m_win=3)
    mod.u_hist = [0.1, 0.2, 0.3]
    assert mod._direct_causal_feedback_at(2) is None


def test_direct_formula_equals_even_odd_hartley_decomposition() -> None:
    mod = _make_modulator(m_win=4)
    mod.u_hist = [0.91, 0.77, 0.72, 0.58, 0.53, 0.49, 0.40, 0.35, 0.31]
    for n in range(mod.m_win, len(mod.u_hist)):
        direct = mod._direct_causal_feedback_at(n)
        decomposed = mod.hartley_decomposition_feedback_at(n)
        assert direct is not None and decomposed is not None
        assert math.isclose(direct, decomposed, rel_tol=1e-10, abs_tol=1e-12)


def test_ema_is_applied_before_bounded_control_map() -> None:
    mod = _make_modulator(m_win=1, alpha=0.5)
    mod.on_batch_end(2.0)
    assert math.isclose(mod.last_ema, 2.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mod.last_u, 2.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)

    mod.on_batch_end(0.0)
    # EMA = 0.5*2 + 0.5*0 = 1; phi(1)=1/2.
    assert math.isclose(mod.last_ema, 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mod.last_u, 0.5, rel_tol=0.0, abs_tol=1e-12)


def test_no_ema_ablation_uses_current_loss_directly() -> None:
    mod = _make_modulator(m_win=1, alpha=0.99, variant="no_ema")
    mod.on_batch_end(2.0)
    mod.on_batch_end(0.0)
    assert math.isclose(mod.last_ema, 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mod.last_u, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_clipped_modulation_never_exceeds_gamma() -> None:
    gamma = 0.05
    mod = _make_modulator(
        m_win=2,
        alpha=0.0,
        gamma=gamma,
        trend_conf_tau=0.0,
        variance_normalize=False,
        use_auto_beta=False,
        beta_fixed=100.0,
        mod_warmup_steps=0,
    )
    losses = [3.0, 2.5, 2.0, 1.4, 1.0, 0.7, 0.5, 0.4, 0.3]
    for loss in losses:
        mod.on_batch_end(loss)
    assert abs(mod.last_delta) <= gamma + 1e-12
    assert mod.last_mod_lr > 0.0


def test_theoretical_kernel_bound_holds() -> None:
    mod = _make_modulator(m_win=5)
    mod.u_hist = [0.05, 0.15, 0.25, 0.45, 0.60, 0.90]
    raw = mod._direct_causal_feedback_at(5)
    assert raw is not None
    bound = 2.0 * mod.causal_kernel_norm * max(abs(v) for v in mod.u_hist)
    assert abs(raw) <= bound + 1e-12


def test_no_hc_ablation_uses_simple_causal_trend() -> None:
    mod = _make_modulator(m_win=3, variant="no_hc")
    mod.u_hist = [0.8, 0.7]
    trend = mod._simple_causal_trend()
    assert trend is not None
    assert trend > 0.0
