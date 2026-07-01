from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(device: str | None = None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_peak_gpu_memory() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**3)
    except Exception:
        return None
    return None


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    state: dict[str, float] = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - state["start"]


def normalize_chrom(chrom: str | int | None) -> str | None:
    if chrom is None:
        return None
    chrom = str(chrom)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def ensure_square_matrix(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix. Got shape={arr.shape}.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    return arr.astype(np.float32)

