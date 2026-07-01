from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray, mode: str = "average") -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if mode == "average":
        return (0.5 * (arr + arr.T)).astype(np.float32)
    if mode == "max":
        return np.maximum(arr, arr.T).astype(np.float32)
    raise ValueError("mode must be 'average' or 'max'.")


def clip_nonnegative(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32).copy()
    arr[arr < 0] = 0.0
    return arr


def distance_zscore_normalize(
    matrix: np.ndarray,
    max_diag: int = 500,
    min_valid: int = 20,
    eps: float = 1e-6,
    keep_outside: bool = False,
) -> tuple[np.ndarray, dict[int, dict[str, float | int]]]:
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Input matrix must be square. Got shape={mat.shape}.")
    n = mat.shape[0]
    max_diag = min(int(max_diag), n - 1)
    mat_z = mat.copy() if keep_outside else np.full_like(mat, np.nan, dtype=np.float32)
    diag_stats: dict[int, dict[str, float | int]] = {}
    np.fill_diagonal(mat_z, np.nan)
    for k in range(1, max_diag + 1):
        vals = np.diagonal(mat, offset=k).astype(np.float32)
        valid = np.isfinite(vals)
        vals_valid = vals[valid]
        if vals_valid.size < min_valid:
            diag_stats[k] = {"mean": float("nan"), "std": float("nan"), "valid_count": int(vals_valid.size)}
            continue
        mu = float(np.nanmean(vals_valid))
        sigma = float(np.nanstd(vals_valid))
        diag_stats[k] = {"mean": mu, "std": sigma, "valid_count": int(vals_valid.size)}
        z_vals = (vals - mu) / (sigma + eps)
        i = np.arange(0, n - k)
        j = i + k
        mat_z[i, j] = z_vals
        mat_z[j, i] = z_vals
    return mat_z.astype(np.float32), diag_stats


def distance_zscore_denormalize(
    matrix_z: np.ndarray,
    diag_stats: dict[int, dict[str, float | int]],
    max_diag: int = 500,
    fill_diagonal: float = np.nan,
    clip_nonnegative_values: bool = True,
) -> np.ndarray:
    mat_z = np.asarray(matrix_z, dtype=np.float32)
    n = mat_z.shape[0]
    max_diag = min(int(max_diag), n - 1)
    out = np.full_like(mat_z, np.nan, dtype=np.float32)
    for k in range(1, max_diag + 1):
        stats = diag_stats.get(k) or diag_stats.get(str(k))
        if not stats:
            continue
        mu = float(stats.get("mean", np.nan))
        sigma = float(stats.get("std", np.nan))
        if not np.isfinite(mu) or not np.isfinite(sigma):
            continue
        vals = np.diagonal(mat_z, offset=k).astype(np.float32) * sigma + mu
        if clip_nonnegative_values:
            vals = np.maximum(vals, 0.0)
        i = np.arange(0, n - k)
        j = i + k
        out[i, j] = vals
        out[j, i] = vals
    np.fill_diagonal(out, fill_diagonal)
    return out.astype(np.float32)

