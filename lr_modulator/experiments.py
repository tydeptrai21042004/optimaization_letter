from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import torch
import torch.optim as optim

from .config import ExperimentConfig
from .data import build_loaders, recommended_input_size
from .engine import fit
from .io_utils import history_path, load_json, make_label, save_history_csv, save_json, summary_path
from .model_zoo import build_model
from .runtime import set_seed
from .schedulers import Controller


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
    )
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
    tr_loader, va_loader, te_loader, num_classes = build_loaders(config, device, dataset, input_size, batch_size, seed)
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
        if config.skip_on_fail:
            print(f"[FAIL->SKIP] {args} | reason: {repr(exc)}")
        else:
            raise


def run_all(config: ExperimentConfig, device: torch.device) -> List[Dict]:
    all_summaries: List[Dict] = []

    for ds in config.scratch_datasets:
        for model_name in config.scratch_models:
            for method in config.scratch_methods:
                for seed in config.seeds:
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

