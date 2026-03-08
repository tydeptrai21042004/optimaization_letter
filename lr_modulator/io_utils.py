from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List


def make_label(
    dataset: str,
    model_name: str,
    method: str,
    seed: int,
    alpha: float,
    gamma: float,
    use_auto_beta: bool,
    pretrained: bool,
) -> str:
    label = f"{dataset}__{model_name}__{method}__seed{seed}"
    if method in {"ours_cosine", "ours_onecycle"}:
        label += f"__a{alpha:.2f}_g{gamma:.2f}_auto{int(use_auto_beta)}"
    if pretrained:
        label += "__pretrained"
    return label


def summary_path(save_dir: str, label: str) -> str:
    return os.path.join(save_dir, f"{label}_summary.json")


def history_path(save_dir: str, label: str) -> str:
    return os.path.join(save_dir, f"{label}_history.csv")


def checkpoint_path(save_dir: str, label: str, kind: str = "latest") -> str:
    return os.path.join(save_dir, f"{label}_{kind}.pt")


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_aggregate_csv(path: str, summaries: Iterable[Dict]) -> None:
    summaries = list(summaries)
    if not summaries:
        return
    keys = sorted(set().union(*[s.keys() for s in summaries]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)