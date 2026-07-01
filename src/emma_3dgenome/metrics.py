from __future__ import annotations

import numpy as np


def _masked_vectors(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(true, dtype=np.float32)
    y_pred = np.asarray(pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"true and pred must have the same shape. Got {y_true.shape} and {y_pred.shape}.")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != y_true.shape:
            raise ValueError(f"mask shape must match matrix shape. Got {mask_arr.shape} and {y_true.shape}.")
        valid &= mask_arr
    if not np.any(valid):
        raise ValueError("No valid points for metric calculation.")
    return y_true[valid].astype(np.float64), y_pred[valid].astype(np.float64)


def masked_mse(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y_true, y_pred = _masked_vectors(true, pred, mask)
    return float(np.mean((y_true - y_pred) ** 2))


def masked_mae(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y_true, y_pred = _masked_vectors(true, pred, mask)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_on_mask(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y_true, y_pred = _masked_vectors(true, pred, mask)
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_on_mask(true: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    try:
        from scipy.stats import spearmanr
    except Exception as exc:
        raise ImportError("spearman_on_mask requires scipy.") from exc

    y_true, y_pred = _masked_vectors(true, pred, mask)
    if y_true.size < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).correlation)
