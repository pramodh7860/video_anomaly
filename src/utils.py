"""
Shared utilities: config loading, seeding, device selection.
"""
import random
import yaml
import numpy as np
import torch


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def moving_average(scores: np.ndarray, window: int) -> np.ndarray:
    """Simple causal moving average used to smooth snippet anomaly scores
    for the online inference stream (uses only past+current values so it
    stays valid for sequential/online use)."""
    if window <= 1:
        return scores
    out = np.zeros_like(scores)
    for i in range(len(scores)):
        lo = max(0, i - window + 1)
        out[i] = scores[lo : i + 1].mean()
    return out
