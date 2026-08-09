from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np


def seed_all(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed python/random, numpy, and torch (if installed).
    deterministic=True makes cudnn deterministic (slower, but repeatable).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # optional

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Usually faster
            torch.backends.cudnn.benchmark = True
    except Exception:
        # torch not installed or not available; ok for prepare_data
        pass


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return it as a Python dict.
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    """
    Save a dict to YAML.
    """
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
