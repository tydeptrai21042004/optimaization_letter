from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


SUPPORTED_METHODS = {
    "constant",
    "step",
    "cosine",
    "onecycle",
    "warmup_cosine",
    "warm_restarts",
    "plateau",
    "ours_cosine",
    "ours_onecycle",
    "ours_warmup_cosine",
}


def get_device() -> Tuple[torch.device, bool]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA: {name}")
        return torch.device("cuda"), True

    print("[Device] CPU (will be slow)")
    return torch.device("cpu"), False


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def validate_method(method: str) -> str:
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: {method}. "
            f"Supported methods are: {sorted(SUPPORTED_METHODS)}"
        )
    return method


def validate_methods(methods: Iterable[str]) -> List[str]:
    return [validate_method(m) for m in methods]


def config_to_dict(config: Any) -> Dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("config_to_dict expects a dataclass instance.")
    return asdict(config)
