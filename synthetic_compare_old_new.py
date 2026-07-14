from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def phi(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return x / (c + x + 1e-12)


def ema(loss: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(loss, dtype=float)
    for t, value in enumerate(loss):
        out[t] = value if t == 0 else alpha * out[t - 1] + (1.0 - alpha) * value
    return out


def causal_weights(m_win: int, rho: float) -> np.ndarray:
    weights = np.asarray([rho ** (m - 1) for m in range(1, m_win + 1)], dtype=float)
    weights /= weights.sum()
    return weights


def causal_feedback(u: np.ndarray, m_win: int, rho: float) -> np.ndarray:
    """Implementation form of (q_M^e *_H u) + (q_M^o *_H R u)."""
    weights = causal_weights(m_win, rho)
    raw = np.zeros_like(u, dtype=float)
    for t in range(m_win, len(u)):
        raw[t] = sum(weights[m - 1] * (u[t - m] - u[t]) for m in range(1, m_win + 1))
    return raw


def corrected_ema_hartley_feedback(
    loss: np.ndarray,
    *,
    alpha: float = 0.95,
    m_win: int = 3,
    rho: float = 0.8,
    var_alpha: float = 0.95,
    tau: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = ema(loss, alpha)
    u = phi(smoothed)
    raw = causal_feedback(u, m_win=m_win, rho=rho)

    score = np.zeros_like(raw)
    active = np.zeros_like(raw, dtype=bool)
    variance = 0.0
    for t in range(1, len(u)):
        du = u[t] - u[t - 1]
        variance = var_alpha * variance + (1.0 - var_alpha) * du * du
        if t < m_win:
            continue
        score[t] = raw[t] / (np.sqrt(max(variance, 0.0)) + 1e-8)
        active[t] = abs(score[t]) >= tau
    return raw, score, active


def make_loss(seed: int, T: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    clean = np.empty(T, dtype=float)
    clean[:220] = 2.0 - 0.0030 * t[:220]
    clean[220:360] = clean[219] + 0.00005 * (t[220:360] - 220)
    clean[360:] = clean[359] - 0.0017 * (t[360:] - 360)
    clean += 0.035 * np.sin(2.0 * np.pi * t / 55.0)

    noise = rng.normal(0.0, 0.055, size=T)
    outlier_mask = rng.random(T) < 0.035
    outliers = outlier_mask * rng.gamma(shape=2.0, scale=0.10, size=T)
    observed = np.maximum(clean + noise + outliers, 0.02)

    true_trend = np.zeros(T, dtype=float)
    true_trend[1:] = clean[:-1] - clean[1:]
    return clean, observed, true_trend


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def main() -> None:
    seeds = list(range(200))
    m_win = 3
    alpha = 0.95
    rho = 0.8
    tau = 0.15

    no_ema_accuracy: list[float] = []
    corrected_precision: list[float] = []
    corrected_coverage: list[float] = []
    corrected_false_active: list[float] = []
    corrected_active_rate: list[float] = []
    max_bound_ratio: list[float] = []

    for seed in seeds:
        _, observed, true_trend = make_loss(seed)
        no_ema_raw = causal_feedback(phi(observed), m_win=m_win, rho=rho)
        corrected_raw, corrected_score, active = corrected_ema_hartley_feedback(
            observed,
            alpha=alpha,
            m_win=m_win,
            rho=rho,
            tau=tau,
        )

        valid = np.arange(len(observed)) >= max(30, m_win)
        nonflat = (np.abs(true_trend) > 7.5e-4) & valid
        flat = (np.abs(true_trend) <= 7.5e-4) & valid

        no_ema_accuracy.append(float(np.mean(np.sign(no_ema_raw[nonflat]) == np.sign(true_trend[nonflat]))))
        active_nonflat = nonflat & active
        corrected_precision.append(
            float(np.mean(np.sign(corrected_raw[active_nonflat]) == np.sign(true_trend[active_nonflat])))
            if active_nonflat.any()
            else 0.0
        )
        corrected_coverage.append(float(np.mean(active[nonflat])))
        corrected_false_active.append(float(np.mean(active[flat])))
        corrected_active_rate.append(float(np.mean(active[valid])))

        # For bounded phi, ||u||_infty <= 1 and normalized kernel mass is one.
        # The paper's conservative bound is |raw_t| <= 2 ||u||_infty.
        u = phi(ema(observed, alpha))
        denominator = 2.0 * max(float(np.max(np.abs(u))), 1e-12)
        max_bound_ratio.append(float(np.max(np.abs(corrected_raw)) / denominator))

    payload = {
        "num_seeds": len(seeds),
        "method": "causal_ema_h_hartley",
        "m_win": m_win,
        "alpha": alpha,
        "rho": rho,
        "hc_delay": 0,
        "trend_conf_tau": tau,
        "no_ema_direction_accuracy": summarize(no_ema_accuracy),
        "corrected_direction_precision_when_active": summarize(corrected_precision),
        "corrected_nonflat_coverage": summarize(corrected_coverage),
        "corrected_false_active_rate_flat": summarize(corrected_false_active),
        "corrected_active_rate": summarize(corrected_active_rate),
        "max_ratio_to_conservative_kernel_bound": summarize(max_bound_ratio),
    }
    Path("synthetic_compare_output.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
