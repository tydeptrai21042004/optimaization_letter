from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch


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
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
