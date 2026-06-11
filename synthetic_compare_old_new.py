from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def phi(x: np.ndarray, c: float = 1.0) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return x / (c + x + 1e-12)


def old_ema_feedback(loss: np.ndarray, alpha: float = 0.90, m_win: int = 3, rho: float = 0.8) -> np.ndarray:
    """Old EMA-causal trend signal used as a baseline."""
    u = phi(loss)
    ema = np.zeros_like(u)
    for t in range(len(u)):
        ema[t] = u[t] if t == 0 else alpha * ema[t - 1] + (1.0 - alpha) * u[t]

    weights = np.array([rho ** (m - 1) for m in range(1, m_win + 1)], dtype=float)
    weights /= weights.sum()
    raw = np.zeros_like(u)
    for t in range(m_win, len(u)):
        raw[t] = sum(weights[m - 1] * (ema[t - m] - ema[t]) for m in range(1, m_win + 1))
    return raw


def hc_kernel(m_win: int, rho: float = 0.8, h: float = 1.0) -> dict[int, float]:
    raw = {m: rho ** abs(m) for m in range(-m_win, m_win + 1)}
    total = sum(raw.values()) + 1e-12
    return {m: w / (2.0 * h * total) for m, w in raw.items()}


def hc_conv(u: np.ndarray, n: int, kernel: dict[int, float], h: float = 1.0):
    if n < 0:
        return None
    last = len(u) - 1
    acc = 0.0
    for m, k in kernel.items():
        idxs = (n - m - 1, n - m + 1, n + m + 1, n + m - 1)
        if min(idxs) < 0 or max(idxs) > last:
            return None
        acc += k * sum(float(u[j]) for j in idxs)
    return 0.5 * h * acc


def new_hc_feedback(
    loss: np.ndarray,
    m_win: int = 3,
    rho: float = 0.8,
    var_alpha: float = 0.95,
    tau: float = 0.25,
    h: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """New delayed HC-convolution score and active gate."""
    u = phi(loss)
    D = m_win + 1
    kernel = hc_kernel(m_win=m_win, rho=rho, h=h)
    score = np.zeros_like(u)
    active = np.zeros_like(u, dtype=bool)
    v = 0.0
    for t in range(1, len(u)):
        du = u[t] - u[t - 1]
        v = var_alpha * v + (1.0 - var_alpha) * (du * du)
        if t < 2 * m_win + 3:
            continue
        z_now = hc_conv(u[: t + 1], t - D, kernel, h=h)
        z_prev = hc_conv(u[: t + 1], t - 1 - D, kernel, h=h)
        if z_now is None or z_prev is None:
            continue
        q = z_prev - z_now
        s = q / (np.sqrt(max(v, 0.0)) + 1e-8)
        score[t] = s
        active[t] = abs(s) >= tau
    return score, active


def make_loss(seed: int, T: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    clean = np.empty(T, dtype=float)
    clean[:220] = 2.0 - 0.0030 * t[:220]
    clean[220:360] = clean[219] + 0.00005 * (t[220:360] - 220)  # nearly flat
    clean[360:] = clean[359] - 0.0017 * (t[360:] - 360)
    clean += 0.035 * np.sin(2.0 * np.pi * t / 55.0)

    noise = rng.normal(0.0, 0.055, size=T)
    outlier_mask = rng.random(T) < 0.035
    outliers = outlier_mask * rng.gamma(shape=2.0, scale=0.10, size=T)
    observed = np.maximum(clean + noise + outliers, 0.02)

    true_trend = np.zeros(T, dtype=float)
    true_trend[1:] = clean[:-1] - clean[1:]
    return clean, observed, true_trend


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def main() -> None:
    seeds = list(range(200))
    old_acc = []
    new_precision = []
    new_coverage = []
    old_false_active = []
    new_false_active = []
    old_abs = []
    new_abs_gated = []
    new_active_rate = []

    m_win = 3
    delay = m_win + 1
    tau = 0.15

    for seed in seeds:
        clean, observed, true_trend = make_loss(seed)
        old = old_ema_feedback(observed, m_win=m_win)
        new, active = new_hc_feedback(observed, m_win=m_win, tau=tau)

        # HC feedback z_t is evaluated at t-D, so compare to the delayed true trend.
        aligned_trend = np.zeros_like(true_trend)
        aligned_trend[delay:] = true_trend[:-delay]

        valid = np.arange(len(observed)) > 30
        nonflat = (np.abs(aligned_trend) > 7.5e-4) & valid
        flat = (np.abs(aligned_trend) <= 7.5e-4) & valid

        old_acc.append(np.mean(np.sign(old[nonflat]) == np.sign(aligned_trend[nonflat])))

        active_nonflat = nonflat & active
        if active_nonflat.sum() > 0:
            new_precision.append(np.mean(np.sign(new[active_nonflat]) == np.sign(aligned_trend[active_nonflat])))
        else:
            new_precision.append(0.0)
        new_coverage.append(np.mean(active[nonflat]))

        # Old EMA has no confidence gate, so every nonzero response is treated as active.
        old_false_active.append(float(np.mean(np.abs(old[flat]) > 1e-12)))
        new_false_active.append(float(np.mean(active[flat])))
        old_abs.append(float(np.mean(np.abs(old[valid]))))
        new_abs_gated.append(float(np.mean(np.abs(new[valid]) * active[valid])))
        new_active_rate.append(float(np.mean(active[valid])))

    payload = {
        "num_seeds": len(seeds),
        "m_win": m_win,
        "hc_delay": delay,
        "trend_conf_tau": tau,
        "old_ema_direction_accuracy_all_responses": summarize(old_acc),
        "new_hc_direction_precision_when_active": summarize(new_precision),
        "new_hc_nonflat_coverage": summarize(new_coverage),
        "old_ema_false_active_rate_flat": summarize(old_false_active),
        "new_hc_false_active_rate_flat": summarize(new_false_active),
        "old_ema_mean_abs_signal": summarize(old_abs),
        "new_hc_mean_abs_signal_gated": summarize(new_abs_gated),
        "new_hc_active_rate_all_valid_steps": summarize(new_active_rate),
    }
    Path("synthetic_compare_output.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
