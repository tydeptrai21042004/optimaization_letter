from __future__ import annotations

import csv
import math
import os
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import ExperimentConfig
from .data import build_loaders, recommended_input_size
from .engine import fit
from .io_utils import (
    batch_history_path,
    history_path,
    load_json,
    make_label,
    save_aggregate_csv,
    save_history_csv,
    save_json,
    summary_path,
)
from .model_zoo import build_model
from .optimizers import build_optimizer_for_method
from .runtime import set_seed
from .schedulers import Controller


def _cfg_get(config: ExperimentConfig, *names: str, default=None):
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return default


def _uniq_keep_order(xs: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _print_run_header(
    task: str,
    dataset: str,
    model_name: str,
    method: str,
    seed: int,
    epochs: int,
    batch_size: int,
    base_lr: float,
    pretrained: bool,
) -> None:
    print(
        "\n[RUN] "
        f"task={task} | dataset={dataset} | model={model_name} | method={method} | "
        f"seed={seed} | epochs={epochs} | batch_size={batch_size} | "
        f"lr={base_lr} | pretrained={pretrained}"
    )


def is_skippable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    known_patterns = [
        "torchvision could not be imported",
        "torchvision models could not be imported",
        "download",
        "http error",
        "urlopen error",
        "temporary failure",
        "no space left on device",
    ]
    return any(p in msg for p in known_patterns)


def methods_for_task(config: ExperimentConfig, task: str, dataset: str) -> List[str]:
    if task == "scratch":
        methods = list(config.scratch_methods)
        if dataset == "cifar10" and config.extra_baselines_cifar10:
            methods.extend(config.extra_baselines_cifar10)
        return _uniq_keep_order(methods)

    if task == "finetune":
        return _uniq_keep_order(list(config.finetune_methods))

    raise ValueError(f"Unknown task: {task}")


def _paired_stats(values_a: List[float], values_b: List[float]) -> Tuple[float, float]:
    """Return (mean_diff, approximate paired t-test p-value).

    Uses scipy if available; otherwise returns p=nan.  This avoids making scipy a
    hard runtime dependency for training.
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float("nan"), float("nan")
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    mean_diff = float(diff.mean())
    try:
        from scipy.stats import ttest_rel  # type: ignore

        p = float(ttest_rel(values_a, values_b).pvalue)
    except Exception:
        p = float("nan")
    return mean_diff, p


def summarize_replicates(summaries: Sequence[Dict], out_path: Optional[str] = None) -> List[Dict]:
    """Aggregate seed replicates with mean/std/95% CI.

    Groups by task, dataset, model, method, and pretrained flag.  Output is useful
    for paper tables and reviewer-requested confidence intervals.
    """
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for s in summaries:
        key = (
            s.get("task"),
            s.get("dataset"),
            s.get("model"),
            s.get("method"),
            s.get("pretrained"),
        )
        grouped[key].append(s)

    rows: List[Dict] = []
    metrics = ["best_val_acc", "test_acc", "test_loss", "time_sec", "clip_rate", "delta_mean_abs_final"]
    for key, items in grouped.items():
        task, dataset, model_name, method, pretrained = key
        row: Dict[str, object] = {
            "task": task,
            "dataset": dataset,
            "model": model_name,
            "method": method,
            "pretrained": pretrained,
            "n_seeds": len(items),
            "seeds": ",".join(str(x.get("seed")) for x in items),
        }
        for metric in metrics:
            vals = [float(x[metric]) for x in items if metric in x and x[metric] is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_ci95"] = ci95
        rows.append(row)  # type: ignore[arg-type]

    if out_path is not None:
        save_aggregate_csv(out_path, rows)
    return rows


def paired_method_tests(
    summaries: Sequence[Dict],
    comparisons: Optional[Sequence[Tuple[str, str]]] = None,
    metric: str = "test_acc",
    out_path: Optional[str] = None,
) -> List[Dict]:
    """Paired seed tests for selected method comparisons.

    By default tests the proposed EMA variants against their base/random rivals.
    """
    if comparisons is None:
        comparisons = [
            ("ours_cosine", "cosine"),
            ("ours_cosine", "random_cosine"),
            ("ours_onecycle", "onecycle"),
            ("ours_onecycle", "random_onecycle"),
            ("ours_warmup_cosine", "warmup_cosine"),
            ("ours_no_ema_cosine", "ours_cosine"),
            ("ours_no_kernel_cosine", "ours_cosine"),
        ]

    # group by task/dataset/model/pretrained, then by method/seed
    contexts: Dict[Tuple, Dict[Tuple[str, int], Dict]] = defaultdict(dict)
    for s in summaries:
        key = (s.get("task"), s.get("dataset"), s.get("model"), s.get("pretrained"))
        contexts[key][(s.get("method"), int(s.get("seed", -1)))] = s

    rows: List[Dict] = []
    for ctx, item_map in contexts.items():
        task, dataset, model_name, pretrained = ctx
        seeds = sorted({seed for (_, seed) in item_map.keys()})
        for method_a, method_b in comparisons:
            vals_a: List[float] = []
            vals_b: List[float] = []
            used_seeds: List[int] = []
            for seed in seeds:
                a = item_map.get((method_a, seed))
                b = item_map.get((method_b, seed))
                if a is None or b is None or metric not in a or metric not in b:
                    continue
                vals_a.append(float(a[metric]))
                vals_b.append(float(b[metric]))
                used_seeds.append(seed)
            if len(vals_a) < 2:
                continue
            mean_diff, p = _paired_stats(vals_a, vals_b)
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "model": model_name,
                    "pretrained": pretrained,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_a": float(np.mean(vals_a)),
                    "mean_b": float(np.mean(vals_b)),
                    "mean_diff_a_minus_b": mean_diff,
                    "paired_t_pvalue": p,
                    "n_pairs": len(vals_a),
                    "seeds": ",".join(map(str, used_seeds)),
                }
            )

    if out_path is not None:
        save_aggregate_csv(out_path, rows)
    return rows


def run_one(
    config: ExperimentConfig,
    device: torch.device,
    dataset: str,
    model_name: str,
    method: str,
    seed: int,
    epochs: int,
    batch_size: int,
    base_lr: float,
    pretrained: bool,
) -> Optional[Dict]:
    label = make_label(
        dataset=dataset,
        model_name=model_name,
        method=method,
        seed=seed,
        alpha=config.alpha,
        gamma=config.gamma,
        use_auto_beta=config.use_auto_beta,
        pretrained=pretrained,
        m_win=config.m_win,
        rho=config.rho,
        beta_fixed=config.beta_fixed,
        relative_trend=config.relative_trend,
    )

    task_tag = "finetune" if pretrained else "scratch"
    lr_tag = f"{base_lr:.8g}".replace(".", "p")
    label = f"{label}_{task_tag}_ep{epochs}_bs{batch_size}_lr{lr_tag}"

    sum_path = summary_path(config.save_dir, label)
    hist_path = history_path(config.save_dir, label)
    batch_hist_path = batch_history_path(config.save_dir, label)

    if config.skip_if_exists and os.path.exists(sum_path):
        print(f"[SKIP] {label}")
        return load_json(sum_path)

    if config.should_stop():
        print("[STOP] Time budget reached (before run).")
        return None

    set_seed(seed, config.deterministic)

    input_size = recommended_input_size(model_name, dataset, pretrained)
    tr_loader, va_loader, te_loader, num_classes = build_loaders(
        config=config,
        device=device,
        dataset=dataset,
        input_size=input_size,
        batch_size=batch_size,
        seed=seed,
    )

    model = build_model(model_name, num_classes, pretrained, input_size).to(device)

    optimizer, optimizer_impl = build_optimizer_for_method(
        method=method,
        params=model.parameters(),
        base_lr=base_lr,
        config=config,
    )

    total_steps = epochs * len(tr_loader)
    controller = Controller(
        optimizer=optimizer,
        config=config,
        method=method,
        total_steps=total_steps,
        steps_per_epoch=len(tr_loader),
        base_lr=base_lr,
        min_lr=config.min_lr,
    )

    t0 = time.time()
    final_metrics, history, batch_history = fit(
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        test_loader=te_loader,
        optimizer=optimizer,
        controller=controller,
        device=device,
        config=config,
        epochs=epochs,
    )
    elapsed = time.time() - t0

    summary = {
        "label": label,
        "dataset": dataset,
        "model": model_name,
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "base_lr": base_lr,
        "pretrained": pretrained,
        "task": task_tag,
        "input_size": input_size,
        "deterministic": config.deterministic,
        "num_workers": config.num_workers,
        "optimizer_impl": optimizer_impl,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "min_lr": config.min_lr,
        "max_lr_factor": config.max_lr_factor,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "m_win": config.m_win,
        "rho": config.rho,
        "bias_correct_ema": config.bias_correct_ema,
        "relative_trend": config.relative_trend,
        "dead_zone_tau": config.dead_zone_tau,
        "variance_normalize": config.variance_normalize,
        "use_ema": config.use_ema,
        "use_kernel": config.use_kernel,
        "use_clipping": config.use_clipping,
        "random_delta_gamma": config.random_delta_gamma,
        "l4_alpha": config.l4_alpha,
        "l4_gamma": config.l4_gamma,
        "hyper_lr": config.hyper_lr,
        "mod_warmup_steps": _cfg_get(config, "mod_warmup_steps", "warmup_steps", default=0),
        "sched_warmup_steps": _cfg_get(config, "sched_warmup_steps", "warmup_steps", default=0),
        "warmup_start_factor": _cfg_get(config, "warmup_start_factor", default=0.1),
        "restart_t0_steps": _cfg_get(config, "restart_t0_steps", "restart_t0", default=10),
        "restart_t_mult": _cfg_get(config, "restart_t_mult", "t_mult", default=1),
        "plateau_mode": _cfg_get(config, "plateau_mode", default="min"),
        "plateau_factor": _cfg_get(config, "plateau_factor", default=0.5),
        "plateau_patience": _cfg_get(config, "plateau_patience", default=2),
        "plateau_threshold": _cfg_get(config, "plateau_threshold", default=1e-4),
        "plateau_threshold_mode": _cfg_get(config, "plateau_threshold_mode", default="rel"),
        "plateau_cooldown": _cfg_get(config, "plateau_cooldown", default=0),
        "use_auto_beta": config.use_auto_beta,
        "beta_fixed": config.beta_fixed,
        "target_mean_abs_delta": config.target_mean_abs_delta,
        "beta_cap": config.beta_cap,
        **final_metrics,
        "time_sec": float(elapsed),
        "time_left_sec": float(config.time_left_sec()),
    }
    summary.update(controller.stats())

    save_json(sum_path, summary)
    save_history_csv(hist_path, history)
    save_history_csv(batch_hist_path, batch_history)
    return summary


def safe_run(
    all_summaries: List[Dict],
    config: ExperimentConfig,
    device: torch.device,
    *args,
) -> None:
    try:
        summary = run_one(config, device, *args)
        if summary is not None:
            all_summaries.append(summary)
    except Exception as exc:
        err_msg = f"[FAIL] {args} | reason: {repr(exc)}"
        print(err_msg)

        log_path = os.path.join(config.save_dir, "failed_runs.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(err_msg + "\n")
            f.write(traceback.format_exc() + "\n")

        if config.skip_on_fail and is_skippable_error(exc):
            print("[FAIL->SKIP] known environment/data issue")
        else:
            raise


def run_method_suite(
    config: ExperimentConfig,
    device: torch.device,
    task: str,
    dataset: str,
    model_name: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    base_lr: Optional[float] = None,
    seeds: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
) -> List[Dict]:
    all_summaries: List[Dict] = []

    if task == "scratch":
        pretrained = False
        epochs = config.scratch_epochs if epochs is None else epochs
        batch_size = config.scratch_batch if batch_size is None else batch_size
        base_lr = config.lr_scratch if base_lr is None else base_lr
    elif task == "finetune":
        pretrained = True
        epochs = config.finetune_epochs if epochs is None else epochs
        batch_size = config.finetune_batch if batch_size is None else batch_size
        base_lr = config.lr_finetune if base_lr is None else base_lr
    else:
        raise ValueError("task must be either 'scratch' or 'finetune'")

    if seeds is None:
        seeds = config.seeds

    methods = list(methods) if methods is not None else methods_for_task(config, task, dataset)

    print("\n" + "=" * 120)
    print(f"Running suite | task={task} | dataset={dataset} | model={model_name}")
    print(f"epochs={epochs} | batch_size={batch_size} | lr={base_lr} | pretrained={pretrained}")
    print(f"methods={methods} | seeds={list(seeds)}")
    print("=" * 120)

    for method in methods:
        for seed in seeds:
            _print_run_header(
                task=task,
                dataset=dataset,
                model_name=model_name,
                method=method,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                base_lr=base_lr,
                pretrained=pretrained,
            )
            safe_run(
                all_summaries,
                config,
                device,
                dataset,
                model_name,
                method,
                seed,
                epochs,
                batch_size,
                base_lr,
                pretrained,
            )
            if config.should_stop():
                break
        if config.should_stop():
            break

    return all_summaries


def run_ablation_suite(
    config: ExperimentConfig,
    device: torch.device,
    task: str,
    dataset: str,
    model_name: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    base_lr: Optional[float] = None,
    seeds: Optional[Sequence[int]] = None,
) -> List[Dict]:
    return run_method_suite(
        config=config,
        device=device,
        task=task,
        dataset=dataset,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        base_lr=base_lr,
        seeds=seeds,
        methods=config.ablation_methods,
    )


def run_all(config: ExperimentConfig, device: torch.device) -> List[Dict]:
    all_summaries: List[Dict] = []

    for ds in config.scratch_datasets:
        for model_name in config.scratch_models:
            methods = methods_for_task(config, "scratch", ds)
            for method in methods:
                for seed in config.seeds:
                    _print_run_header(
                        task="scratch",
                        dataset=ds,
                        model_name=model_name,
                        method=method,
                        seed=seed,
                        epochs=config.scratch_epochs,
                        batch_size=config.scratch_batch,
                        base_lr=config.lr_scratch,
                        pretrained=False,
                    )
                    safe_run(
                        all_summaries,
                        config,
                        device,
                        ds,
                        model_name,
                        method,
                        seed,
                        config.scratch_epochs,
                        config.scratch_batch,
                        config.lr_scratch,
                        False,
                    )
                    if config.should_stop():
                        break
                if config.should_stop():
                    break
            if config.should_stop():
                break
        if config.should_stop():
            break

    if config.do_finetune and not config.should_stop():
        for ds in config.finetune_datasets:
            for model_name in config.finetune_models:
                methods = methods_for_task(config, "finetune", ds)
                for method in methods:
                    for seed in config.seeds:
                        _print_run_header(
                            task="finetune",
                            dataset=ds,
                            model_name=model_name,
                            method=method,
                            seed=seed,
                            epochs=config.finetune_epochs,
                            batch_size=config.finetune_batch,
                            base_lr=config.lr_finetune,
                            pretrained=True,
                        )
                        safe_run(
                            all_summaries,
                            config,
                            device,
                            ds,
                            model_name,
                            method,
                            seed,
                            config.finetune_epochs,
                            config.finetune_batch,
                            config.lr_finetune,
                            True,
                        )
                        if config.should_stop():
                            break
                    if config.should_stop():
                        break
                if config.should_stop():
                    break
            if config.should_stop():
                break

    return all_summaries
