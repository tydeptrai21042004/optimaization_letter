from __future__ import annotations

import os
import time
import traceback
from typing import Dict, List, Optional, Sequence

import torch
import torch.optim as optim

from .config import ExperimentConfig
from .data import build_loaders, recommended_input_size
from .engine import fit
from .io_utils import history_path, load_json, make_label, save_history_csv, save_json, summary_path
from .model_zoo import build_model
from .runtime import set_seed
from .schedulers import Controller


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
        f"seed={seed} | epochs={epochs} | batch_size={batch_size} | lr={base_lr} | pretrained={pretrained}"
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
    base_label = make_label(
        dataset=dataset,
        model_name=model_name,
        method=method,
        seed=seed,
        alpha=config.alpha,
        gamma=config.gamma,
        use_auto_beta=config.use_auto_beta,
        pretrained=pretrained,
    )

    task_tag = "finetune" if pretrained else "scratch"
    lr_tag = f"{base_lr:.8g}".replace(".", "p")
    label = f"{base_label}_{task_tag}_ep{epochs}_bs{batch_size}_lr{lr_tag}"

    sum_path = summary_path(config.save_dir, label)
    hist_path = history_path(config.save_dir, label)

    if config.skip_if_exists and os.path.exists(sum_path):
        print(f"[SKIP] {label}")
        return load_json(sum_path)

    if config.should_stop():
        print("[STOP] Time budget reached (before run).")
        return None

    set_seed(seed, config.deterministic)

    input_size = recommended_input_size(model_name, dataset, pretrained)
    tr_loader, va_loader, te_loader, num_classes = build_loaders(
        config, device, dataset, input_size, batch_size, seed
    )

    model = build_model(model_name, num_classes, pretrained, input_size).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
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
    final_metrics, history = fit(
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
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "min_lr": config.min_lr,
        "alpha": config.alpha,
        "gamma": config.gamma,
        "m_win": config.m_win,
        "rho": config.rho,
        "warmup_steps": config.warmup_steps,
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
    return summary


def safe_run(all_summaries: List[Dict], config: ExperimentConfig, device: torch.device, *args) -> None:
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


def run_all(config: ExperimentConfig, device: torch.device) -> List[Dict]:
    all_summaries: List[Dict] = []

    for ds in config.scratch_datasets:
        for model_name in config.scratch_models:
            for method in config.scratch_methods:
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

    if config.extra_baselines_cifar10 and not config.should_stop():
        for model_name in config.scratch_models:
            for method in config.extra_baselines_cifar10:
                for seed in config.seeds:
                    _print_run_header(
                        task="scratch",
                        dataset="cifar10",
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
                        "cifar10",
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

    if config.do_finetune and not config.should_stop():
        for ds in config.finetune_datasets:
            for model_name in config.finetune_models:
                for method in config.finetune_methods:
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