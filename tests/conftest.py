from __future__ import annotations

import torch

# Keep CPU-only validation fast and deterministic in constrained notebook/CI/Kaggle environments.
torch.set_num_threads(1)
