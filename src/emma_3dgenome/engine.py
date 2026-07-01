# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np

from PyEMD import EMD
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Reproducibility and CUDA
# ============================================================

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_cuda_memory(tag=""):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[CUDA {tag}] "
        f"allocated={allocated:.3f} GB, "
        f"reserved={reserved:.3f} GB, "
        f"max_allocated={max_allocated:.3f} GB"
    )


# ============================================================
# 1. Basic utilities
# ============================================================

def init_1d_signal_for_emd(y, method="linear"):
    y = np.asarray(y, dtype=np.float32)
    out = y.copy()

    n = len(y)
    x = np.arange(n)
    valid = np.isfinite(y)

    if valid.sum() == 0:
        return np.zeros_like(y, dtype=np.float32)

    if valid.sum() == 1:
        out[~valid] = y[valid][0]
        return out.astype(np.float32)

    if method == "zero":
        out[~valid] = 0.0

    elif method == "mean":
        out[~valid] = np.nanmean(y)

    elif method == "linear":
        f = interp1d(
            x[valid],
            y[valid],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        out[~valid] = f(x[~valid])

    else:
        raise ValueError(f"Unknown init method: {method}")

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def smooth_1d_nan_aware(y, window=7):
    y = np.asarray(y, dtype=np.float32)

    if window <= 1:
        return y.astype(np.float32)

    valid = np.isfinite(y)
    y0 = np.where(valid, y, 0.0).astype(np.float32)
    w = valid.astype(np.float32)

    kernel = np.ones(int(window), dtype=np.float32)
    kernel = kernel / kernel.sum()

    num = np.convolve(y0, kernel, mode="same")
    den = np.convolve(w, kernel, mode="same")

    out = num / (den + 1e-6)
    out[den < 1e-6] = np.nan

    return out.astype(np.float32)


def nan_gaussian_filter_2d(mat, sigma=1.0):
    mat = np.asarray(mat, dtype=np.float32)

    valid = np.isfinite(mat)
    x0 = np.where(valid, mat, 0.0).astype(np.float32)
    w = valid.astype(np.float32)

    num = gaussian_filter(x0, sigma=sigma)
    den = gaussian_filter(w, sigma=sigma)

    out = num / (den + 1e-6)
    out[den < 1e-6] = np.nan

    return out.astype(np.float32)


def robust_zscore_1d(x, eps=1e-6, clip_value=8.0):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    valid = np.isfinite(x)

    if valid.sum() < 5:
        return np.zeros_like(x, dtype=np.float32), 0.0, 1.0

    med = np.nanmedian(x[valid])
    mad = np.nanmedian(np.abs(x[valid] - med))
    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-3:
        scale = np.nanstd(x[valid])
        if not np.isfinite(scale) or scale < 1e-3:
            scale = 1.0

    out = (x - med) / (scale + eps)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -clip_value, clip_value)

    return out.astype(np.float32), float(med), float(scale)


def robust_clip_by_observed(y_pred, y_obs, sigma=5.0):
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_obs = np.asarray(y_obs, dtype=np.float32)

    out = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    obs = y_obs[np.isfinite(y_obs)]

    if len(obs) >= 20:
        med = np.nanmedian(obs)
        mad = np.nanmedian(np.abs(obs - med)) + 1e-6
        rsd = 1.4826 * mad

        if np.isfinite(med) and np.isfinite(rsd) and rsd > 1e-6:
            lo = med - sigma * rsd
            hi = med + sigma * rsd
            out = np.clip(out, lo, hi)

    return out.astype(np.float32)


def robust_clip_imfs_by_observed(pred_imfs, obs_imfs, sigma=5.0):
    pred_imfs = np.asarray(pred_imfs, dtype=np.float32)
    obs_imfs = np.asarray(obs_imfs, dtype=np.float32)

    out = pred_imfs.copy()

    for m in range(out.shape[0]):
        obs = obs_imfs[m]
        obs = obs[np.isfinite(obs)]

        if len(obs) < 20:
            continue

        med = np.nanmedian(obs)
        mad = np.nanmedian(np.abs(obs - med)) + 1e-6
        rsd = 1.4826 * mad

        if np.isfinite(med) and np.isfinite(rsd) and rsd > 1e-6:
            lo = med - sigma * rsd
            hi = med + sigma * rsd
            out[m] = np.clip(out[m], lo, hi)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def emd_decompose_1d_fixed(y_init, max_imfs=4):
    y_init = np.asarray(y_init, dtype=np.float32)
    y_init = np.nan_to_num(y_init, nan=0.0, posinf=0.0, neginf=0.0)

    emd = EMD()
    imfs = emd.emd(y_init)

    if imfs is None or len(imfs) == 0:
        imfs_fixed = np.zeros((max_imfs, len(y_init)), dtype=np.float32)
        residual = y_init.copy().astype(np.float32)
        return imfs_fixed, residual, 0

    imfs = np.asarray(imfs, dtype=np.float32)
    imfs = np.nan_to_num(imfs, nan=0.0, posinf=0.0, neginf=0.0)

    raw_imf_num = imfs.shape[0]

    if raw_imf_num >= max_imfs:
        imfs_fixed = imfs[:max_imfs]
        residual = y_init - np.sum(imfs_fixed, axis=0)
    else:
        pad = np.zeros((max_imfs - raw_imf_num, len(y_init)), dtype=np.float32)
        imfs_fixed = np.vstack([imfs, pad])
        residual = y_init - np.sum(imfs, axis=0)

    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    return imfs_fixed.astype(np.float32), residual.astype(np.float32), int(raw_imf_num)


# ============================================================
# 2. Context initialization before EMD
# ============================================================

def build_rowcol_context_init_matrix(
    mat_z_missing,
    mask_block,
    selected_bins=None,
    max_iter=30,
    sigma=1.2,
    rowcol_weight=0.65,
    local_weight=0.35,
    neighbor_window=80,
    neighbor_tau=30.0,
    min_ref_rows=3,
    keep_observed=True,
    verbose=True,
):
    mat_z_missing = np.asarray(mat_z_missing, dtype=np.float32)
    mask_block = np.asarray(mask_block, dtype=bool)

    if mat_z_missing.shape[0] != mat_z_missing.shape[1]:
        raise ValueError(f"mat_z_missing must be square, got {mat_z_missing.shape}")

    if mask_block.shape != mat_z_missing.shape:
        raise ValueError("mask_block shape must match mat_z_missing")

    n = mat_z_missing.shape[0]

    if selected_bins is None:
        selected_bins = np.where(np.any(mask_block, axis=1))[0]

    selected_bins = np.asarray(selected_bins, dtype=np.int64)
    selected_bins = selected_bins[(selected_bins >= 0) & (selected_bins < n)]
    selected_set = set(selected_bins.tolist())

    obs_vals = mat_z_missing[np.isfinite(mat_z_missing) & (~mask_block)]
    global_mean = float(np.nanmean(obs_vals)) if len(obs_vals) > 0 else 0.0

    if verbose:
        print(f"[Init] selected bins={len(selected_bins)}, global_mean={global_mean:.5f}")

    local_prior = mat_z_missing.copy()

    for it in range(max_iter):
        nan_target = mask_block & (~np.isfinite(local_prior))

        if nan_target.sum() == 0:
            break

        sm = nan_gaussian_filter_2d(local_prior, sigma=sigma)
        can_fill = nan_target & np.isfinite(sm)
        local_prior[can_fill] = sm[can_fill]

        if verbose and (it == 0 or (it + 1) % 10 == 0):
            print(
                f"[Init local context] iter={it + 1}, "
                f"remain_nan={np.sum(mask_block & (~np.isfinite(local_prior)))}"
            )

    local_prior[~np.isfinite(local_prior)] = global_mean

    rowcol_prior = local_prior.copy()

    valid_ref_bins = []
    for r in range(n):
        if r in selected_set:
            continue

        row = mat_z_missing[r, :]
        col = mat_z_missing[:, r]

        row_valid = np.isfinite(row).sum()
        col_valid = np.isfinite(col).sum()

        if max(row_valid, col_valid) >= max(20, int(0.05 * n)):
            valid_ref_bins.append(r)

    valid_ref_bins = np.asarray(valid_ref_bins, dtype=np.int64)

    if verbose:
        print(f"[Init row/col context] valid reference bins={len(valid_ref_bins)}")

    for count, b in enumerate(selected_bins):
        b = int(b)

        if len(valid_ref_bins) == 0:
            continue

        left = max(0, b - neighbor_window)
        right = min(n, b + neighbor_window + 1)

        local_refs = valid_ref_bins[
            (valid_ref_bins >= left)
            & (valid_ref_bins < right)
            & (valid_ref_bins != b)
        ]

        if len(local_refs) < min_ref_rows:
            dist_all = np.abs(valid_ref_bins - b)
            order = np.argsort(dist_all)
            refs = valid_ref_bins[order[: min(20, len(order))]]
        else:
            dist_local = np.abs(local_refs - b)
            order = np.argsort(dist_local)
            refs = local_refs[order[: min(20, len(order))]]

        if len(refs) == 0:
            continue

        weights = np.exp(-np.abs(refs - b) / max(neighbor_tau, 1e-6)).astype(np.float32)
        weights = weights / (weights.sum() + 1e-6)

        row_profiles = mat_z_missing[refs, :]
        row_valid = np.isfinite(row_profiles)
        row_profiles0 = np.where(row_valid, row_profiles, 0.0)

        row_num = np.sum(row_profiles0 * weights[:, None], axis=0)
        row_den = np.sum(row_valid.astype(np.float32) * weights[:, None], axis=0)

        row_prior = row_num / (row_den + 1e-6)
        row_prior[row_den < 1e-6] = np.nan

        target_row = mask_block[b, :] & np.isfinite(row_prior)
        rowcol_prior[b, target_row] = row_prior[target_row]

        col_profiles = mat_z_missing[:, refs].T
        col_valid = np.isfinite(col_profiles)
        col_profiles0 = np.where(col_valid, col_profiles, 0.0)

        col_num = np.sum(col_profiles0 * weights[:, None], axis=0)
        col_den = np.sum(col_valid.astype(np.float32) * weights[:, None], axis=0)

        col_prior = col_num / (col_den + 1e-6)
        col_prior[col_den < 1e-6] = np.nan

        target_col = mask_block[:, b] & np.isfinite(col_prior)
        rowcol_prior[target_col, b] = col_prior[target_col]

        if verbose and (count < 10 or count % 20 == 0):
            print(
                f"[Init row/col context] bin={b}, "
                f"refs={len(refs)}, "
                f"row_filled={target_row.sum()}, "
                f"col_filled={target_col.sum()}"
            )

    mat_init = mat_z_missing.copy()

    target = mask_block

    both = target & np.isfinite(rowcol_prior) & np.isfinite(local_prior)
    rowcol_only = target & np.isfinite(rowcol_prior) & (~np.isfinite(local_prior))
    local_only = target & np.isfinite(local_prior) & (~np.isfinite(rowcol_prior))

    mat_init[both] = (
        rowcol_weight * rowcol_prior[both]
        + local_weight * local_prior[both]
    )
    mat_init[rowcol_only] = rowcol_prior[rowcol_only]
    mat_init[local_only] = local_prior[local_only]

    still_nan = target & (~np.isfinite(mat_init))
    mat_init[still_nan] = global_mean

    if keep_observed:
        obs = np.isfinite(mat_z_missing) & (~mask_block)
        mat_init[obs] = mat_z_missing[obs]

    mat_init = 0.5 * (mat_init + mat_init.T)
    np.fill_diagonal(mat_init, np.nan)

    if verbose:
        print(
            f"[Init done] finite in mask={np.isfinite(mat_init[mask_block]).sum()} / {mask_block.sum()}"
        )

    return mat_init.astype(np.float32)


# ============================================================
# 3. Diagonal EMD decomposition
# ============================================================

def decompose_one_diagonal_to_imf_features(
    y_missing,
    y_mask,
    y_init_external=None,
    max_imfs=4,
    residual_smooth_window=11,
    init_method="linear",
):
    y_missing = np.asarray(y_missing, dtype=np.float32)
    y_mask = np.asarray(y_mask, dtype=bool)

    valid_obs = np.isfinite(y_missing) & (~y_mask)
    missing = y_mask

    if y_init_external is None:
        y_init = init_1d_signal_for_emd(y_missing, method=init_method)
    else:
        y_init = np.asarray(y_init_external, dtype=np.float32).copy()

        obs = np.isfinite(y_missing) & (~y_mask)
        y_init[obs] = y_missing[obs]

        if np.any(~np.isfinite(y_init)):
            fallback = init_1d_signal_for_emd(y_missing, method="linear")
            bad = ~np.isfinite(y_init)
            y_init[bad] = fallback[bad]

        y_init = np.nan_to_num(y_init, nan=0.0, posinf=0.0, neginf=0.0)

    imfs, residual, raw_imf_num = emd_decompose_1d_fixed(
        y_init=y_init,
        max_imfs=max_imfs,
    )

    residual_smooth = smooth_1d_nan_aware(residual, window=residual_smooth_window)
    residual_smooth = np.where(np.isfinite(residual_smooth), residual_smooth, residual)
    residual_smooth = np.nan_to_num(residual_smooth, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "y_missing": y_missing.astype(np.float32),
        "y_init": y_init.astype(np.float32),
        "imfs": imfs.astype(np.float32),
        "residual": residual.astype(np.float32),
        "residual_smooth": residual_smooth.astype(np.float32),
        "valid_obs": valid_obs.astype(bool),
        "missing": missing.astype(bool),
        "raw_imf_num": int(raw_imf_num),
        "length": int(len(y_missing)),
    }


def build_all_diagonal_imf_features(
    mat_z_missing,
    mask_block,
    mat_init=None,
    max_diag=500,
    min_diag=1,
    max_imfs=4,
    residual_smooth_window=11,
    verbose=True,
):
    mat_z_missing = np.asarray(mat_z_missing, dtype=np.float32)
    mask_block = np.asarray(mask_block, dtype=bool)

    if mat_init is not None:
        mat_init = np.asarray(mat_init, dtype=np.float32)
        if mat_init.shape != mat_z_missing.shape:
            raise ValueError("mat_init shape must match mat_z_missing")

    if mat_z_missing.shape[0] != mat_z_missing.shape[1]:
        raise ValueError(f"mat_z_missing must be square, got {mat_z_missing.shape}")

    n = mat_z_missing.shape[0]
    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    diag_features = {}
    t0 = time.perf_counter()

    for k in range(min_diag, max_diag + 1):
        i = np.arange(0, n - k)
        j = i + k

        y_missing = mat_z_missing[i, j].astype(np.float32)
        y_mask = mask_block[i, j].astype(bool)

        y_init_external = None if mat_init is None else mat_init[i, j].astype(np.float32)

        feat = decompose_one_diagonal_to_imf_features(
            y_missing=y_missing,
            y_mask=y_mask,
            y_init_external=y_init_external,
            max_imfs=max_imfs,
            residual_smooth_window=residual_smooth_window,
            init_method="linear",
        )

        feat["diag"] = int(k)
        diag_features[int(k)] = feat

        if verbose and (k <= 10 or k % 50 == 0):
            print(
                f"[EMD diag {k:4d}] "
                f"len={len(y_missing)}, "
                f"obs={feat['valid_obs'].sum()}, "
                f"real_mask={feat['missing'].sum()}, "
                f"raw_imfs={feat['raw_imf_num']}"
            )

    if verbose:
        print(f"[Build IMF features] done. time={time.perf_counter() - t0:.2f}s")

    return diag_features


# ============================================================
# 4. IMF reconstruction
# ============================================================

def reconstruct_signal_from_imfs(
    imfs,
    residual,
    imf_weights=(0.08, 1.35, 1.20, 0.90),
    residual_weight=1.0,
    residual_smooth_window=11,
):
    imfs = np.asarray(imfs, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)

    max_imfs = imfs.shape[0]

    imf_weights = np.asarray(imf_weights, dtype=np.float32)

    if len(imf_weights) < max_imfs:
        pad = np.ones(max_imfs - len(imf_weights), dtype=np.float32)
        imf_weights = np.concatenate([imf_weights, pad])

    residual_sm = smooth_1d_nan_aware(residual, window=residual_smooth_window)
    residual_sm = np.where(np.isfinite(residual_sm), residual_sm, residual)
    residual_sm = np.nan_to_num(residual_sm, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.sum(imfs * imf_weights[:max_imfs, None], axis=0) + residual_weight * residual_sm
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.astype(np.float32)


def build_matrix_from_reconstructed_diagonals(
    diag_features,
    n,
    max_diag=500,
    min_diag=1,
    use_key="imfs",
    residual_key="residual_smooth",
    imf_weights=(0.08, 1.35, 1.20, 0.90),
    residual_weight=1.0,
    residual_smooth_window=11,
):
    out = np.full((n, n), np.nan, dtype=np.float32)
    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    for k in range(min_diag, max_diag + 1):
        if k not in diag_features:
            continue

        feat = diag_features[k]

        y_rec = reconstruct_signal_from_imfs(
            imfs=feat[use_key],
            residual=feat[residual_key],
            imf_weights=imf_weights,
            residual_weight=residual_weight,
            residual_smooth_window=residual_smooth_window,
        )

        i = np.arange(0, n - k)
        j = i + k

        out[i, j] = y_rec
        out[j, i] = y_rec

    np.fill_diagonal(out, np.nan)
    return out.astype(np.float32)


# ============================================================
# 5. Masked IMF AutoEncoder Dataset
# ============================================================

class MaskedIMFAutoEncoderDataset(Dataset):
    """
    训练逻辑：
    1. 真实缺失区域 real_missing 不参与训练。
    2. 从正常观测区域随机 pseudo-mask。
    3. 输入：pseudo-mask 后的 IMF 表征。
    4. 目标：恢复 pseudo-mask 位置的 target_imfs 和 target_y。
    """

    def __init__(
        self,
        diag_features,
        patch_len=256,
        n_samples=30000,
        pseudo_mask_ratio=0.15,
        min_obs_in_patch=32,
        max_imfs=4,
        max_diag=500,
        residual_smooth_window=11,
        seed=2026,
        input_clip=8.0,
        target_clip=5.0,
    ):
        self.diag_features = diag_features
        self.patch_len = int(patch_len)
        self.n_samples = int(n_samples)
        self.pseudo_mask_ratio = float(pseudo_mask_ratio)
        self.min_obs_in_patch = int(min_obs_in_patch)
        self.max_imfs = int(max_imfs)
        self.max_diag = int(max_diag)
        self.residual_smooth_window = int(residual_smooth_window)
        self.input_clip = float(input_clip)
        self.target_clip = float(target_clip)

        self.rng = np.random.default_rng(seed)

        self.available_diags = []

        for k, feat in diag_features.items():
            valid_obs = feat["valid_obs"]
            if feat["length"] >= 16 and valid_obs.sum() >= self.min_obs_in_patch:
                self.available_diags.append(int(k))

        if len(self.available_diags) == 0:
            raise ValueError("No available diagonals for MAE training.")

    def __len__(self):
        return self.n_samples

    def _sample_patch_position(self, L):
        if L <= self.patch_len:
            return 0, L

        start = self.rng.integers(0, L - self.patch_len + 1)
        end = start + self.patch_len
        return int(start), int(end)

    def _pad(self, x, start, end, fill=0.0):
        p = x[start:end]
        out = np.full(self.patch_len, fill, dtype=np.float32)
        out[:len(p)] = p
        valid_len = len(p)
        return out, valid_len

    def __getitem__(self, idx):
        for _ in range(80):
            k = int(self.rng.choice(self.available_diags))
            feat = self.diag_features[k]
            L = feat["length"]

            start, end = self._sample_patch_position(L)

            y_original, valid_len = self._pad(feat["y_missing"], start, end, fill=np.nan)
            y_context_init, _ = self._pad(feat["y_init"], start, end, fill=0.0)
            real_missing, _ = self._pad(feat["missing"].astype(np.float32), start, end, fill=0.0)
            target_residual, _ = self._pad(feat["residual_smooth"], start, end, fill=0.0)

            target_imfs = np.zeros((self.max_imfs, self.patch_len), dtype=np.float32)
            for m in range(self.max_imfs):
                target_imfs[m], _ = self._pad(feat["imfs"][m], start, end, fill=0.0)

            valid_len_mask = np.zeros(self.patch_len, dtype=bool)
            valid_len_mask[:valid_len] = True

            train_obs = (
                np.isfinite(y_original)
                & (real_missing < 0.5)
                & valid_len_mask
            )

            obs_idx = np.where(train_obs)[0]

            if len(obs_idx) < self.min_obs_in_patch:
                continue

            n_mask = max(4, int(len(obs_idx) * self.pseudo_mask_ratio))
            n_mask = min(n_mask, len(obs_idx))

            pseudo_idx = self.rng.choice(obs_idx, size=n_mask, replace=False)

            pseudo_mask = np.zeros(self.patch_len, dtype=np.float32)
            pseudo_mask[pseudo_idx] = 1.0

            y_pseudo_missing = y_original.copy()
            y_pseudo_missing[pseudo_idx] = np.nan

            y_pseudo_init = y_context_init.copy()
            obs_patch = np.isfinite(y_pseudo_missing)
            y_pseudo_init[obs_patch] = y_pseudo_missing[obs_patch]

            bad = ~np.isfinite(y_pseudo_init)
            if np.any(bad):
                fallback = init_1d_signal_for_emd(y_pseudo_missing, method="linear")
                y_pseudo_init[bad] = fallback[bad]

            y_pseudo_init = np.nan_to_num(
                y_pseudo_init,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            input_imfs, input_residual, _ = emd_decompose_1d_fixed(
                y_init=y_pseudo_init,
                max_imfs=self.max_imfs,
            )

            input_residual_smooth = smooth_1d_nan_aware(
                input_residual,
                window=self.residual_smooth_window,
            )
            input_residual_smooth = np.where(
                np.isfinite(input_residual_smooth),
                input_residual_smooth,
                input_residual,
            )
            input_residual_smooth = np.nan_to_num(
                input_residual_smooth,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            y_input = y_pseudo_missing.copy()
            y_input[~np.isfinite(y_input)] = y_pseudo_init[~np.isfinite(y_input)]
            y_input[~np.isfinite(y_input)] = 0.0

            y_input_z, _, _ = robust_zscore_1d(y_input, clip_value=self.input_clip)
            y_init_z, _, _ = robust_zscore_1d(y_pseudo_init, clip_value=self.input_clip)
            residual_z, _, _ = robust_zscore_1d(input_residual_smooth, clip_value=self.input_clip)

            imf_z_list = []
            for m in range(self.max_imfs):
                imf_z, _, _ = robust_zscore_1d(input_imfs[m], clip_value=self.input_clip)
                imf_z_list.append(imf_z)

            dist_value = np.log1p(float(k)) / np.log1p(float(self.max_diag))
            dist_channel = np.full(self.patch_len, dist_value, dtype=np.float32)

            valid_channel = valid_len_mask.astype(np.float32)

            channels = [
                y_input_z,
                y_init_z,
            ]
            channels.extend(imf_z_list)
            channels.extend([
                residual_z,
                real_missing.astype(np.float32),
                pseudo_mask.astype(np.float32),
                dist_channel,
                valid_channel,
            ])

            x = np.stack(channels, axis=0).astype(np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.clip(x, -self.input_clip, self.input_clip).astype(np.float32)

            target_y = np.zeros(self.patch_len, dtype=np.float32)
            target_y[pseudo_idx] = y_original[pseudo_idx]
            target_y = np.nan_to_num(target_y, nan=0.0, posinf=0.0, neginf=0.0)
            target_y = np.clip(target_y, -self.target_clip, self.target_clip)

            return {
                "x": torch.from_numpy(x),
                "input_imfs": torch.from_numpy(input_imfs.astype(np.float32)),
                "input_residual": torch.from_numpy(input_residual_smooth[None, :].astype(np.float32)),
                "target_imfs": torch.from_numpy(target_imfs.astype(np.float32)),
                "target_residual": torch.from_numpy(target_residual[None, :].astype(np.float32)),
                "target_y": torch.from_numpy(target_y[None, :].astype(np.float32)),
                "loss_mask": torch.from_numpy(pseudo_mask[None, :].astype(np.float32)),
            }

        in_channels = 2 + self.max_imfs + 5

        return {
            "x": torch.zeros((in_channels, self.patch_len), dtype=torch.float32),
            "input_imfs": torch.zeros((self.max_imfs, self.patch_len), dtype=torch.float32),
            "input_residual": torch.zeros((1, self.patch_len), dtype=torch.float32),
            "target_imfs": torch.zeros((self.max_imfs, self.patch_len), dtype=torch.float32),
            "target_residual": torch.zeros((1, self.patch_len), dtype=torch.float32),
            "target_y": torch.zeros((1, self.patch_len), dtype=torch.float32),
            "loss_mask": torch.zeros((1, self.patch_len), dtype=torch.float32),
        }


# ============================================================
# 6. Masked IMF AutoEncoder model
# ============================================================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        groups = min(8, out_ch)

        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class MaskedIMFAutoEncoder1D(nn.Module):
    """
    输入:
        pseudo-masked IMF features

    输出:
        reconstructed IMF coefficients, shape [B, max_imfs, L]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        depth=3,
        dropout=0.05,
    ):
        super().__init__()

        enc_channels = []
        ch = base_channels

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = in_channels

        for _ in range(depth):
            self.enc_blocks.append(ConvBlock1D(prev, ch, dropout=dropout))
            self.downs.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))
            enc_channels.append(ch)
            prev = ch
            ch *= 2

        self.mid = ConvBlock1D(prev, ch, dropout=dropout)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for d in reversed(range(depth)):
            skip_ch = enc_channels[d]
            self.ups.append(
                nn.ConvTranspose1d(ch, skip_ch, kernel_size=4, stride=2, padding=1)
            )
            self.dec_blocks.append(
                ConvBlock1D(skip_ch + skip_ch, skip_ch, dropout=dropout)
            )
            ch = skip_ch

        self.out = nn.Conv1d(ch, out_channels, kernel_size=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        skips = []
        h = x
        input_len = x.shape[-1]

        for enc, down in zip(self.enc_blocks, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.mid(h)

        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)

            if h.shape[-1] > skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            elif h.shape[-1] < skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))

            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        if h.shape[-1] > input_len:
            h = h[..., :input_len]
        elif h.shape[-1] < input_len:
            h = F.pad(h, (0, input_len - h.shape[-1]))

        return self.out(h)


# ============================================================
# 7. Masked IMF AutoEncoder loss
# ============================================================

def masked_imf_autoencoder_loss(
    pred_imfs,
    input_imfs,
    input_residual,
    target_imfs,
    target_y,
    mask,
    imf_weights=(0.08, 1.35, 1.20, 0.90),
    signal_weight=1.0,
    imf_weight=0.25,
    context_weight=0.05,
    smooth_weight=0.001,
):
    pred_imfs = torch.nan_to_num(pred_imfs, nan=0.0, posinf=0.0, neginf=0.0)
    input_imfs = torch.nan_to_num(input_imfs, nan=0.0, posinf=0.0, neginf=0.0)
    input_residual = torch.nan_to_num(input_residual, nan=0.0, posinf=0.0, neginf=0.0)
    target_imfs = torch.nan_to_num(target_imfs, nan=0.0, posinf=0.0, neginf=0.0)
    target_y = torch.nan_to_num(target_y, nan=0.0, posinf=0.0, neginf=0.0)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

    pred_imfs = torch.clamp(pred_imfs, -5.0, 5.0)
    target_imfs = torch.clamp(target_imfs, -5.0, 5.0)
    target_y = torch.clamp(target_y, -5.0, 5.0)

    B, M, L = pred_imfs.shape

    w = torch.tensor(imf_weights, dtype=torch.float32, device=pred_imfs.device)

    if len(w) < M:
        pad = torch.ones(M - len(w), dtype=torch.float32, device=pred_imfs.device)
        w = torch.cat([w, pad], dim=0)

    w = w[:M].view(1, M, 1)

    recon_y = torch.sum(pred_imfs * w, dim=1, keepdim=True) + input_residual
    recon_y = torch.nan_to_num(recon_y, nan=0.0, posinf=0.0, neginf=0.0)

    valid = mask > 0.5

    if valid.sum() == 0:
        return pred_imfs.sum() * 0.0

    signal_loss = F.smooth_l1_loss(recon_y[valid], target_y[valid])

    imf_mask = mask.expand_as(pred_imfs) > 0.5
    imf_loss = F.smooth_l1_loss(pred_imfs[imf_mask], target_imfs[imf_mask])

    context_mask = mask.expand_as(pred_imfs) < 0.5
    if context_mask.sum() > 0:
        context_loss = F.smooth_l1_loss(pred_imfs[context_mask], input_imfs[context_mask])
    else:
        context_loss = pred_imfs.sum() * 0.0

    if pred_imfs.shape[-1] >= 2:
        smooth_loss = torch.mean(torch.abs(pred_imfs[..., 1:] - pred_imfs[..., :-1]))
    else:
        smooth_loss = pred_imfs.sum() * 0.0

    loss = (
        signal_weight * signal_loss
        + imf_weight * imf_loss
        + context_weight * context_loss
        + smooth_weight * smooth_loss
    )

    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

    return loss


# ============================================================
# 8. Train Masked IMF AutoEncoder
# ============================================================

def train_masked_imf_autoencoder(
    diag_features,
    max_diag=500,
    max_imfs=4,
    residual_smooth_window=11,
    patch_len=256,
    n_samples=30000,
    pseudo_mask_ratio=0.15,
    batch_size=128,
    epochs=20,
    lr=1e-4,
    weight_decay=1e-4,
    base_channels=32,
    depth=3,
    dropout=0.05,
    imf_weights=(0.08, 1.35, 1.20, 0.90),
    seed=2026,
    device=None,
    num_workers=4,
    prefetch_factor=2,
    verbose=True,
    debug_batch=True,
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MaskedIMFAutoEncoderDataset(
        diag_features=diag_features,
        patch_len=patch_len,
        n_samples=n_samples,
        pseudo_mask_ratio=pseudo_mask_ratio,
        min_obs_in_patch=32,
        max_imfs=max_imfs,
        max_diag=max_diag,
        residual_smooth_window=residual_smooth_window,
        seed=seed,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)

    in_channels = 2 + max_imfs + 5

    model = MaskedIMFAutoEncoder1D(
        in_channels=in_channels,
        out_channels=max_imfs,
        base_channels=base_channels,
        depth=depth,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    print(
        f"[DataLoader] batch_size={batch_size}, "
        f"num_workers={loader.num_workers}, "
        f"pin_memory={loader.pin_memory}"
    )

    if debug_batch:
        dbg = next(iter(loader))
        print("[Debug MAE train batch]")
        print("x:", dbg["x"].shape, torch.isfinite(dbg["x"]).all().item())
        print("input_imfs:", dbg["input_imfs"].shape, torch.isfinite(dbg["input_imfs"]).all().item())
        print("target_imfs:", dbg["target_imfs"].shape, torch.isfinite(dbg["target_imfs"]).all().item())
        print("target_y:", dbg["target_y"].shape, torch.isfinite(dbg["target_y"]).all().item())
        print("loss_mask sum:", dbg["loss_mask"].sum().item())

    best_loss = np.inf
    best_state = None
    patience = 8
    bad_epochs = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        skipped = 0
        n_batches = 0
        t0 = time.perf_counter()

        for batch in loader:
            n_batches += 1

            x = batch["x"].to(device, non_blocking=True).float()
            input_imfs = batch["input_imfs"].to(device, non_blocking=True).float()
            input_residual = batch["input_residual"].to(device, non_blocking=True).float()
            target_imfs = batch["target_imfs"].to(device, non_blocking=True).float()
            target_y = batch["target_y"].to(device, non_blocking=True).float()
            loss_mask = batch["loss_mask"].to(device, non_blocking=True).float()

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            input_imfs = torch.nan_to_num(input_imfs, nan=0.0, posinf=0.0, neginf=0.0)
            input_residual = torch.nan_to_num(input_residual, nan=0.0, posinf=0.0, neginf=0.0)
            target_imfs = torch.nan_to_num(target_imfs, nan=0.0, posinf=0.0, neginf=0.0)
            target_y = torch.nan_to_num(target_y, nan=0.0, posinf=0.0, neginf=0.0)
            loss_mask = torch.nan_to_num(loss_mask, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad(set_to_none=True)

            pred_imfs = model(x)

            loss = masked_imf_autoencoder_loss(
                pred_imfs=pred_imfs,
                input_imfs=input_imfs,
                input_residual=input_residual,
                target_imfs=target_imfs,
                target_y=target_y,
                mask=loss_mask,
                imf_weights=imf_weights,
                signal_weight=1.0,
                imf_weight=0.25,
                context_weight=0.05,
                smooth_weight=0.001,
            )

            if not torch.isfinite(loss):
                skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        scheduler.step()

        elapsed = time.perf_counter() - t0
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
        batch_per_sec = n_batches / max(elapsed, 1e-6)

        if verbose:
            print(
                f"[Epoch {ep:03d}/{epochs}] "
                f"loss={mean_loss:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                f"skipped={skipped}, "
                f"batches={n_batches}, "
                f"batch/s={batch_per_sec:.2f}, "
                f"time={elapsed:.2f}s"
            )
            print_cuda_memory(f"epoch {ep}")

        if np.isfinite(mean_loss) and mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            if verbose:
                print(f"[Early stop] best_loss={best_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    return model, {
        "best_loss": float(best_loss),
        "device": device,
        "in_channels": int(in_channels),
        "out_channels": int(max_imfs),
        "patch_len": int(patch_len),
        "max_imfs": int(max_imfs),
    }


# ============================================================
# 9. Inference
# ============================================================

def build_real_mask_inference_input_patch(
    feat,
    k,
    start,
    end,
    patch_len,
    max_imfs,
    max_diag,
    input_clip=8.0,
):
    def pad(x, fill=0.0):
        p = x[start:end]
        out = np.full(patch_len, fill, dtype=np.float32)
        out[:len(p)] = p
        valid_len = len(p)
        return out, valid_len

    y_missing, valid_len = pad(feat["y_missing"], fill=np.nan)
    y_init, _ = pad(feat["y_init"], fill=0.0)
    residual, _ = pad(feat["residual_smooth"], fill=0.0)

    imf_z_list = []
    for m in range(max_imfs):
        imf_m, _ = pad(feat["imfs"][m], fill=0.0)
        imf_z, _, _ = robust_zscore_1d(imf_m, clip_value=input_clip)
        imf_z_list.append(imf_z)

    real_missing, _ = pad(feat["missing"].astype(np.float32), fill=0.0)

    valid_len_mask = np.zeros(patch_len, dtype=np.float32)
    valid_len_mask[:valid_len] = 1.0

    y_input = y_missing.copy()
    y_input[~np.isfinite(y_input)] = y_init[~np.isfinite(y_input)]
    y_input[~np.isfinite(y_input)] = 0.0

    y_input_z, _, _ = robust_zscore_1d(y_input, clip_value=input_clip)
    y_init_z, _, _ = robust_zscore_1d(y_init, clip_value=input_clip)
    residual_z, _, _ = robust_zscore_1d(residual, clip_value=input_clip)

    dist_value = np.log1p(float(k)) / np.log1p(float(max_diag))
    dist_channel = np.full(patch_len, dist_value, dtype=np.float32)

    pseudo_mask_channel = np.zeros(patch_len, dtype=np.float32)

    channels = [
        y_input_z,
        y_init_z,
    ]
    channels.extend(imf_z_list)
    channels.extend([
        residual_z,
        real_missing.astype(np.float32),
        pseudo_mask_channel,
        dist_channel,
        valid_len_mask,
    ])

    x = np.stack(channels, axis=0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -input_clip, input_clip).astype(np.float32)

    return x, valid_len


@torch.no_grad()
def predict_one_diagonal_mae_imfs(
    model,
    feat,
    k,
    patch_len=256,
    stride=128,
    max_imfs=4,
    max_diag=500,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    L = feat["length"]

    imfs_sum = np.zeros((max_imfs, L), dtype=np.float32)
    weight_sum = np.zeros((max_imfs, L), dtype=np.float32)

    if L <= patch_len:
        starts = [0]
    else:
        starts = list(range(0, L - patch_len + 1, stride))
        if starts[-1] != L - patch_len:
            starts.append(L - patch_len)

    for start in starts:
        end = min(L, start + patch_len)

        x, valid_len = build_real_mask_inference_input_patch(
            feat=feat,
            k=k,
            start=start,
            end=end,
            patch_len=patch_len,
            max_imfs=max_imfs,
            max_diag=max_diag,
        )

        xt = torch.from_numpy(x[None, :, :]).to(device).float()
        pred_imfs = model(xt).detach().cpu().numpy()[0]

        pred_imfs = np.nan_to_num(pred_imfs, nan=0.0, posinf=0.0, neginf=0.0)
        pred_imfs = np.clip(pred_imfs, -5.0, 5.0).astype(np.float32)

        real_end = start + valid_len

        imfs_sum[:, start:real_end] += pred_imfs[:, :valid_len]
        weight_sum[:, start:real_end] += 1.0

    pred_imfs_full = imfs_sum / (weight_sum + 1e-6)
    pred_imfs_full = np.nan_to_num(pred_imfs_full, nan=0.0, posinf=0.0, neginf=0.0)

    pred_imfs_full = robust_clip_imfs_by_observed(
        pred_imfs=pred_imfs_full,
        obs_imfs=feat["imfs"],
        sigma=5.0,
    )

    return pred_imfs_full.astype(np.float32)


@torch.no_grad()
def apply_masked_imf_autoencoder_to_real_mask(
    model,
    diag_features,
    max_diag=500,
    min_diag=1,
    max_imfs=4,
    patch_len=256,
    stride=128,
    replace_strength=1.0,
    obs_imf_beta=0.0,
    device=None,
    verbose=True,
):
    out_features = {}

    t0 = time.perf_counter()

    for k in range(min_diag, max_diag + 1):
        if k not in diag_features:
            continue

        feat = diag_features[k]
        imfs = feat["imfs"].copy().astype(np.float32)

        missing = feat["missing"].astype(bool)
        valid_obs = feat["valid_obs"].astype(bool)

        pred_imfs = predict_one_diagonal_mae_imfs(
            model=model,
            feat=feat,
            k=k,
            patch_len=patch_len,
            stride=stride,
            max_imfs=max_imfs,
            max_diag=max_diag,
            device=device,
        )

        imfs_new = imfs.copy()

        for m in range(max_imfs):
            imfs_new[m, missing] = (
                (1.0 - replace_strength) * imfs[m, missing]
                + replace_strength * pred_imfs[m, missing]
            )

            if obs_imf_beta > 0:
                imfs_new[m, valid_obs] = (
                    (1.0 - obs_imf_beta) * imfs[m, valid_obs]
                    + obs_imf_beta * pred_imfs[m, valid_obs]
                )

        new_feat = {}
        for key, value in feat.items():
            if isinstance(value, np.ndarray):
                new_feat[key] = value.copy()
            else:
                new_feat[key] = value

        new_feat["imfs_mae"] = imfs_new.astype(np.float32)
        new_feat["imfs_pred"] = pred_imfs.astype(np.float32)

        out_features[k] = new_feat

        if verbose and (k <= 10 or k % 50 == 0):
            print(
                f"[Masked IMF AE real-mask diag {k:4d}] "
                f"len={feat['length']}, "
                f"real_mask={missing.sum()}, "
                f"pred_abs_mean={np.nanmean(np.abs(pred_imfs)):.5f}"
            )

    if verbose:
        print(f"[Apply Masked IMF AE] done. time={time.perf_counter() - t0:.2f}s")

    return out_features


# ============================================================
# 10. Matrix reconstruction and post-processing
# ============================================================

def reconstruct_matrix_from_mae_imfs(
    diag_features_mae,
    mat_z_missing,
    mask_block,
    max_diag=500,
    min_diag=1,
    imf_weights=(0.08, 1.35, 1.20, 0.90),
    residual_weight=1.0,
    residual_smooth_window=11,
    observed_keep=True,
):
    mat_z_missing = np.asarray(mat_z_missing, dtype=np.float32)
    mask_block = np.asarray(mask_block, dtype=bool)

    n = mat_z_missing.shape[0]
    out = mat_z_missing.copy()

    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    for k in range(min_diag, max_diag + 1):
        if k not in diag_features_mae:
            continue

        feat = diag_features_mae[k]
        imfs_use = feat["imfs_mae"] if "imfs_mae" in feat else feat["imfs"]

        y_rec = reconstruct_signal_from_imfs(
            imfs=imfs_use,
            residual=feat["residual_smooth"],
            imf_weights=imf_weights,
            residual_weight=residual_weight,
            residual_smooth_window=residual_smooth_window,
        )

        y_obs = feat["y_missing"]
        y_rec = robust_clip_by_observed(y_rec, y_obs, sigma=5.0)

        i = np.arange(0, n - k)
        j = i + k

        y_final = y_obs.copy()

        missing = mask_block[i, j]
        obs = np.isfinite(y_obs) & (~missing)

        y_final[missing] = y_rec[missing]

        if not observed_keep:
            y_final[obs] = y_rec[obs]
        else:
            y_final[obs] = y_obs[obs]

        out[i, j] = y_final
        out[j, i] = y_final

    obs_all = np.isfinite(mat_z_missing) & (~mask_block)

    if observed_keep:
        out[obs_all] = mat_z_missing[obs_all]

    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, np.nan)

    return out.astype(np.float32)


def diagonal_distribution_calibration(
    mat_cur,
    mat_missing,
    mask,
    max_diag=500,
    min_diag=1,
    strength=0.20,
    eps=1e-6,
):
    mat_cur = np.asarray(mat_cur, dtype=np.float32)
    mat_missing = np.asarray(mat_missing, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    out = mat_cur.copy()
    n = out.shape[0]

    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    for k in range(min_diag, max_diag + 1):
        i = np.arange(0, n - k)
        j = i + k

        obs = np.isfinite(mat_missing[i, j]) & (~mask[i, j])
        fill = mask[i, j] & np.isfinite(out[i, j])

        if obs.sum() < 20 or fill.sum() < 5:
            continue

        obs_vals = mat_missing[i[obs], j[obs]]
        fill_vals = out[i[fill], j[fill]]

        mu_obs = np.nanmean(obs_vals)
        sd_obs = np.nanstd(obs_vals)

        mu_fill = np.nanmean(fill_vals)
        sd_fill = np.nanstd(fill_vals)

        vals_cal = (fill_vals - mu_fill) / (sd_fill + eps) * (sd_obs + eps) + mu_obs
        vals_new = (1.0 - strength) * fill_vals + strength * vals_cal

        out[i[fill], j[fill]] = vals_new
        out[j[fill], i[fill]] = vals_new

    obs_all = np.isfinite(mat_missing) & (~mask)
    out[obs_all] = mat_missing[obs_all]

    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, np.nan)

    return out.astype(np.float32)


def suppress_diagonal_streaks_in_mask(
    mat_cur,
    mat_missing,
    mask,
    max_diag=500,
    min_diag=1,
    diag_smooth_window=9,
    strength=0.04,
):
    mat_cur = np.asarray(mat_cur, dtype=np.float32)
    mat_missing = np.asarray(mat_missing, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    out = mat_cur.copy()
    n = out.shape[0]
    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    for k in range(min_diag, max_diag + 1):
        i = np.arange(0, n - k)
        j = i + k

        vals = out[i, j].astype(np.float32)
        target = mask[i, j] & np.isfinite(vals)

        if target.sum() < 5:
            continue

        vals_smooth = smooth_1d_nan_aware(vals, window=diag_smooth_window)
        valid = target & np.isfinite(vals_smooth)

        vals_new = vals.copy()
        vals_new[valid] = (
            (1.0 - strength) * vals[valid]
            + strength * vals_smooth[valid]
        )

        out[i, j] = vals_new
        out[j, i] = vals_new

    obs_all = np.isfinite(mat_missing) & (~mask)
    out[obs_all] = mat_missing[obs_all]

    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, np.nan)

    return out.astype(np.float32)


# ============================================================
# 11. Evaluation
# ============================================================

def evaluate_imputation_basic(
    mat_z_true,
    mat_z_pred,
    eval_mask,
    max_diag=500,
    min_diag=1,
):
    mat_z_true = np.asarray(mat_z_true, dtype=np.float32)
    mat_z_pred = np.asarray(mat_z_pred, dtype=np.float32)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    n = mat_z_true.shape[0]
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])

    mask_eval = (
        eval_mask
        & (dist >= min_diag)
        & (dist <= max_diag)
        & np.isfinite(mat_z_true)
        & np.isfinite(mat_z_pred)
    )

    y_true = mat_z_true[mask_eval]
    y_pred = mat_z_pred[mask_eval]

    if len(y_true) == 0:
        raise ValueError("No valid points for evaluation.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) >= 2 else np.nan
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) >= 2 else np.nan

    return {
        "valid_points": int(len(y_true)),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


def compute_scc_by_distance(
    mat_true,
    mat_pred,
    max_diag=500,
    min_diag=1,
    min_points=20,
):
    mat_true = np.asarray(mat_true, dtype=np.float32)
    mat_pred = np.asarray(mat_pred, dtype=np.float32)

    n = mat_true.shape[0]
    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    cors = []
    weights = []

    for k in range(min_diag, max_diag + 1):
        i = np.arange(0, n - k)
        j = i + k

        x = mat_true[i, j]
        y = mat_pred[i, j]

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() < min_points:
            continue

        r = pearsonr(x[valid], y[valid])[0]
        if np.isfinite(r):
            cors.append(r)
            weights.append(valid.sum())

    if len(cors) == 0:
        return np.nan

    cors = np.asarray(cors, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-6)

    return float(np.sum(cors * weights))


def compute_insulation_score_simple(mat, window=10):
    mat = np.asarray(mat, dtype=np.float32)
    n = mat.shape[0]

    scores = np.full(n, np.nan, dtype=np.float32)

    for b in range(window, n - window):
        block = mat[b - window:b, b:b + window]
        vals = block[np.isfinite(block)]
        if len(vals) >= max(5, window):
            scores[b] = np.nanmean(vals)

    return scores.astype(np.float32)


def compute_insulation_correlation(mat_true, mat_pred, window=10):
    s1 = compute_insulation_score_simple(mat_true, window=window)
    s2 = compute_insulation_score_simple(mat_pred, window=window)

    valid = np.isfinite(s1) & np.isfinite(s2)

    if valid.sum() < 10:
        return np.nan

    return float(pearsonr(s1[valid], s2[valid])[0])


def evaluate_hic_imputation_and_structure(
    mat_z_true,
    mat_z_pred,
    eval_mask,
    max_diag=500,
    min_diag=1,
    insulation_windows=(5, 10, 20),
):
    result = evaluate_imputation_basic(
        mat_z_true=mat_z_true,
        mat_z_pred=mat_z_pred,
        eval_mask=eval_mask,
        max_diag=max_diag,
        min_diag=min_diag,
    )

    result["scc_by_distance"] = compute_scc_by_distance(
        mat_true=mat_z_true,
        mat_pred=mat_z_pred,
        max_diag=max_diag,
        min_diag=min_diag,
    )

    for w in insulation_windows:
        result[f"insulation_corr_w{w}"] = compute_insulation_correlation(
            mat_true=mat_z_true,
            mat_pred=mat_z_pred,
            window=w,
        )

    return result


def print_metrics(name, metrics):
    print(f"\n[{name}]")

    if metrics is None:
        print("None")
        return

    for k, v in metrics.items():
        print(f"{k}: {v}")


# ============================================================
# 12. Full pipeline
# ============================================================

def run_context_init_masked_imf_autoencoder_emd_hic_imputation(
    mat_z,
    mat_z_missing,
    mask_block,
    selected_bins=None,

    max_diag=500,
    min_diag=1,
    max_imfs=4,

    # context initialization
    init_max_iter=30,
    init_sigma=1.2,
    init_rowcol_weight=0.65,
    init_local_weight=0.35,
    init_neighbor_window=80,
    init_neighbor_tau=30.0,

    # EMD
    residual_smooth_window=11,

    # Masked IMF AutoEncoder
    patch_len=256,
    stride=128,
    n_samples=30000,
    pseudo_mask_ratio=0.15,
    batch_size=128,
    epochs=20,
    lr=1e-4,
    base_channels=32,
    depth=3,
    dropout=0.05,
    num_workers=4,
    prefetch_factor=2,

    # MAE inference
    replace_strength=1.0,
    obs_imf_beta=0.0,

    # reconstruction
    imf_weights=(0.08, 1.35, 1.20, 0.90),
    residual_weight=1.0,

    # post-processing
    do_streak_suppression=False,
    streak_strength=0.04,
    do_diag_calibration=True,
    diag_calib_strength=0.20,

    seed=2026,
    device=None,
    verbose=True,
    debug_batch=True,
):
    set_seed(seed)

    mat_z = np.asarray(mat_z, dtype=np.float32) if mat_z is not None else None
    mat_z_missing = np.asarray(mat_z_missing, dtype=np.float32)
    mask_block = np.asarray(mask_block, dtype=bool)

    if mat_z_missing.shape[0] != mat_z_missing.shape[1]:
        raise ValueError(f"mat_z_missing must be square, got {mat_z_missing.shape}")

    if mask_block.shape != mat_z_missing.shape:
        raise ValueError("mask_block shape must match mat_z_missing")

    n = mat_z_missing.shape[0]
    max_diag = min(max_diag, n - 1)
    min_diag = max(1, min_diag)

    print("\n[Stage 0] Build row/column context initialization before EMD")
    mat_init = build_rowcol_context_init_matrix(
        mat_z_missing=mat_z_missing,
        mask_block=mask_block,
        selected_bins=selected_bins,
        max_iter=init_max_iter,
        sigma=init_sigma,
        rowcol_weight=init_rowcol_weight,
        local_weight=init_local_weight,
        neighbor_window=init_neighbor_window,
        neighbor_tau=init_neighbor_tau,
        keep_observed=True,
        verbose=verbose,
    )

    print("\n[Stage 1] Distance-stratified EMD decomposition with context initialization")
    diag_features = build_all_diagonal_imf_features(
        mat_z_missing=mat_z_missing,
        mask_block=mask_block,
        mat_init=mat_init,
        max_diag=max_diag,
        min_diag=min_diag,
        max_imfs=max_imfs,
        residual_smooth_window=residual_smooth_window,
        verbose=verbose,
    )

    print("\n[Stage 1.5] Build EMD frequency-reweighted prior")
    mat_emd_prior = build_matrix_from_reconstructed_diagonals(
        diag_features=diag_features,
        n=n,
        max_diag=max_diag,
        min_diag=min_diag,
        use_key="imfs",
        residual_key="residual_smooth",
        imf_weights=imf_weights,
        residual_weight=residual_weight,
        residual_smooth_window=residual_smooth_window,
    )

    mat_emd_basic = mat_z_missing.copy()
    fill_target = mask_block & np.isfinite(mat_emd_prior)
    mat_emd_basic[fill_target] = mat_emd_prior[fill_target]

    obs_all = np.isfinite(mat_z_missing) & (~mask_block)
    mat_emd_basic[obs_all] = mat_z_missing[obs_all]
    mat_emd_basic = 0.5 * (mat_emd_basic + mat_emd_basic.T)
    np.fill_diagonal(mat_emd_basic, np.nan)

    print("\n[Stage 2] Train Masked IMF AutoEncoder with pseudo-mask supervision")
    model, train_report = train_masked_imf_autoencoder(
        diag_features=diag_features,
        max_diag=max_diag,
        max_imfs=max_imfs,
        residual_smooth_window=residual_smooth_window,
        patch_len=patch_len,
        n_samples=n_samples,
        pseudo_mask_ratio=pseudo_mask_ratio,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=1e-4,
        base_channels=base_channels,
        depth=depth,
        dropout=dropout,
        imf_weights=imf_weights,
        seed=seed,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        verbose=verbose,
        debug_batch=debug_batch,
    )

    device_used = train_report["device"]

    print("\n[Stage 3] Apply trained Masked IMF AutoEncoder to real missing IMF coefficients")
    diag_features_mae = apply_masked_imf_autoencoder_to_real_mask(
        model=model,
        diag_features=diag_features,
        max_diag=max_diag,
        min_diag=min_diag,
        max_imfs=max_imfs,
        patch_len=patch_len,
        stride=stride,
        replace_strength=replace_strength,
        obs_imf_beta=obs_imf_beta,
        device=device_used,
        verbose=verbose,
    )

    print("\n[Stage 4] Frequency-aware IMF reconstruction")
    mat_mae_imputed = reconstruct_matrix_from_mae_imfs(
        diag_features_mae=diag_features_mae,
        mat_z_missing=mat_z_missing,
        mask_block=mask_block,
        max_diag=max_diag,
        min_diag=min_diag,
        imf_weights=imf_weights,
        residual_weight=residual_weight,
        residual_smooth_window=residual_smooth_window,
        observed_keep=True,
    )

    mat_final = mat_mae_imputed.copy()

    if do_streak_suppression:
        print("\n[Post 1] Mild diagonal streak suppression")
        mat_final = suppress_diagonal_streaks_in_mask(
            mat_cur=mat_final,
            mat_missing=mat_z_missing,
            mask=mask_block,
            max_diag=max_diag,
            min_diag=min_diag,
            diag_smooth_window=9,
            strength=streak_strength,
        )

    if do_diag_calibration:
        print("\n[Post 2] Distance-stratified distribution calibration")
        mat_final = diagonal_distribution_calibration(
            mat_cur=mat_final,
            mat_missing=mat_z_missing,
            mask=mask_block,
            max_diag=max_diag,
            min_diag=min_diag,
            strength=diag_calib_strength,
        )

    obs_all = np.isfinite(mat_z_missing) & (~mask_block)
    mat_final[obs_all] = mat_z_missing[obs_all]

    mat_final = 0.5 * (mat_final + mat_final.T)
    np.fill_diagonal(mat_final, np.nan)

    metrics_init = None
    metrics_emd = None
    metrics_mae = None
    metrics_final = None

    if mat_z is not None:
        print("\n[Evaluation on real masked region]")

        metrics_init = evaluate_hic_imputation_and_structure(
            mat_z_true=mat_z,
            mat_z_pred=mat_init,
            eval_mask=mask_block,
            max_diag=max_diag,
            min_diag=min_diag,
        )

        metrics_emd = evaluate_hic_imputation_and_structure(
            mat_z_true=mat_z,
            mat_z_pred=mat_emd_basic,
            eval_mask=mask_block,
            max_diag=max_diag,
            min_diag=min_diag,
        )

        metrics_mae = evaluate_hic_imputation_and_structure(
            mat_z_true=mat_z,
            mat_z_pred=mat_mae_imputed,
            eval_mask=mask_block,
            max_diag=max_diag,
            min_diag=min_diag,
        )

        metrics_final = evaluate_hic_imputation_and_structure(
            mat_z_true=mat_z,
            mat_z_pred=mat_final,
            eval_mask=mask_block,
            max_diag=max_diag,
            min_diag=min_diag,
        )

        print_metrics("Context initialization", metrics_init)
        print_metrics("EMD frequency prior", metrics_emd)
        print_metrics("Masked IMF AE imputed", metrics_mae)
        print_metrics("Final", metrics_final)

    reports = {
        "train_report": train_report,
        "metrics_init": metrics_init,
        "metrics_emd": metrics_emd,
        "metrics_mae": metrics_mae,
        "metrics_final": metrics_final,
        "params": {
            "max_diag": max_diag,
            "min_diag": min_diag,
            "max_imfs": max_imfs,
            "init_max_iter": init_max_iter,
            "init_sigma": init_sigma,
            "init_rowcol_weight": init_rowcol_weight,
            "init_local_weight": init_local_weight,
            "init_neighbor_window": init_neighbor_window,
            "init_neighbor_tau": init_neighbor_tau,
            "residual_smooth_window": residual_smooth_window,
            "patch_len": patch_len,
            "stride": stride,
            "n_samples": n_samples,
            "pseudo_mask_ratio": pseudo_mask_ratio,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "base_channels": base_channels,
            "depth": depth,
            "dropout": dropout,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "replace_strength": replace_strength,
            "obs_imf_beta": obs_imf_beta,
            "imf_weights": tuple(imf_weights),
            "residual_weight": residual_weight,
            "do_streak_suppression": do_streak_suppression,
            "streak_strength": streak_strength,
            "do_diag_calibration": do_diag_calibration,
            "diag_calib_strength": diag_calib_strength,
        },
    }

    return {
        "mat_init": mat_init.astype(np.float32),
        "mat_emd_prior": mat_emd_prior.astype(np.float32),
        "mat_emd_basic": mat_emd_basic.astype(np.float32),
        "mat_mae_imputed": mat_mae_imputed.astype(np.float32),
        "mat_final": mat_final.astype(np.float32),
        "diag_features": diag_features,
        "diag_features_mae": diag_features_mae,
        "model": model,
        "reports": reports,
    }


# ============================================================
# 13. Recommended usage
# ============================================================



import argparse
import json
import math
from pathlib import Path

import cooler


def normalize_chrom(chrom):
    return chrom if str(chrom).startswith("chr") else "chr" + str(chrom)


def cooler_uri(mcool, resolution):
    return f"{mcool}::resolutions/{resolution}"


def load_chrom_matrix(mcool, resolution, chrom, bin_limit=None):
    clr = cooler.Cooler(cooler_uri(mcool, resolution))
    chrom = normalize_chrom(chrom)
    start_bin, end_bin = clr.extent(chrom)
    chrom_bins = int(end_bin - start_bin)
    n_bins = chrom_bins if bin_limit is None else min(int(bin_limit), chrom_bins)
    mat = clr.matrix(balance=False)[start_bin : start_bin + n_bins, start_bin : start_bin + n_bins]
    mat = np.asarray(mat, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(mat, mat.T)


def remove_diagonal_effect_zscore(mat, max_diag=500, min_valid=20, eps=1e-6, keep_outside=False):
    mat = np.asarray(mat, dtype=np.float32)
    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"mat must be square, got shape={mat.shape}")
    mat_z = np.full_like(mat, np.nan, dtype=np.float32)
    diag_stats = {}
    if keep_outside:
        mat_z[:] = mat
    max_diag = min(max_diag, n - 1)
    np.fill_diagonal(mat_z, np.nan)
    for k in range(1, max_diag + 1):
        vals = np.diagonal(mat, offset=k).astype(np.float32)
        valid = np.isfinite(vals)
        vals_valid = vals[valid]
        if vals_valid.size < min_valid:
            diag_stats[k] = {"mean": np.nan, "std": np.nan, "valid_count": int(vals_valid.size)}
            continue
        mu = np.nanmean(vals_valid)
        sigma = np.nanstd(vals_valid)
        diag_stats[k] = {"mean": float(mu), "std": float(sigma), "valid_count": int(vals_valid.size)}
        z_vals = (vals - mu) / (sigma + eps)
        i = np.arange(0, n - k)
        j = i + k
        mat_z[i, j] = z_vals
        mat_z[j, i] = z_vals
    return mat_z, diag_stats


def restore_from_diagonal_zscore(mat_z, diag_stats, max_diag=500, fill_diagonal=np.nan, keep_outside=False, outside_value=np.nan, clip_nonnegative=True):
    mat_z = np.asarray(mat_z, dtype=np.float32)
    n = mat_z.shape[0]
    max_diag = min(max_diag, n - 1)
    mat_rec = mat_z.copy() if keep_outside else np.full_like(mat_z, outside_value, dtype=np.float32)
    for k in range(1, max_diag + 1):
        stats = diag_stats.get(k)
        if stats is None:
            continue
        mu = stats.get("mean", np.nan)
        sigma = stats.get("std", np.nan)
        if not np.isfinite(mu) or not np.isfinite(sigma):
            continue
        vals_z = np.diagonal(mat_z, offset=k).astype(np.float32)
        vals = vals_z * sigma + mu
        if clip_nonnegative:
            vals = np.maximum(vals, 0.0)
        i = np.arange(0, n - k)
        j = i + k
        mat_rec[i, j] = vals
        mat_rec[j, i] = vals
    np.fill_diagonal(mat_rec, fill_diagonal)
    return mat_rec.astype(np.float32)


def make_block_row_col_mask(n, block_sizes=(5, 10, 15, 20, 25, 30), gap=300, start_bin=200, max_diag=None):
    mask = np.zeros((n, n), dtype=bool)
    selected = []
    regions = []
    cur = int(start_bin)
    for size in block_sizes:
        end = min(n, cur + int(size))
        if cur >= n:
            break
        bins = np.arange(cur, end, dtype=np.int64)
        if len(bins) > 0:
            selected.append(bins)
            regions.append((cur, end, int(len(bins))))
        cur = end + int(gap)
    selected_bins = np.concatenate(selected) if selected else np.array([], dtype=np.int64)
    if selected_bins.size:
        if max_diag is None:
            mask[selected_bins, :] = True
            mask[:, selected_bins] = True
        else:
            d = int(max_diag)
            for b in selected_bins.tolist():
                lo = max(0, b - d)
                hi = min(n, b + d + 1)
                mask[b, lo:hi] = True
                mask[lo:hi, b] = True
    return mask, selected_bins, regions


def run_chrom(args, chrom):
    chrom = normalize_chrom(chrom)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[Load] {chrom}", flush=True)
    high = load_chrom_matrix(args.mcool, args.resolution, chrom, bin_limit=args.bin_limit)
    n = high.shape[0]
    full_mask, selected_bins, regions = make_block_row_col_mask(
        n,
        block_sizes=args.block_sizes,
        gap=args.gap,
        start_bin=args.start_bin,
        max_diag=args.mask_max_diag,
    )
    work_n = n
    print(f"[Work] {chrom} shape={high.shape} work_n={work_n} mask_points={int(full_mask.sum())}", flush=True)
    high_work = high.astype(np.float32)
    mat_z, diag_stats = remove_diagonal_effect_zscore(high_work, max_diag=args.mask_max_diag, min_valid=20)
    work_mask = full_mask[:work_n, :work_n].copy()
    mat_z_missing = mat_z.copy()
    mat_z_missing[work_mask] = np.nan
    work_selected = selected_bins[selected_bins < work_n]
    result = run_context_init_masked_imf_autoencoder_emd_hic_imputation(
        mat_z=None,
        mat_z_missing=mat_z_missing,
        mask_block=work_mask,
        selected_bins=work_selected,
        max_diag=args.mask_max_diag,
        min_diag=1,
        max_imfs=5,
        pseudo_mask_ratio=args.pseudo_mask_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        base_channels=args.base_channels,
        depth=args.depth,
        patch_len=args.patch_len,
        n_samples=args.n_samples,
        stride=128,
        num_workers=0,
        prefetch_factor=2,
        replace_strength=1.0,
        obs_imf_beta=0.0,
        imf_weights=(0.08, 1.35, 1.20, 1.90, 0.80),
        residual_weight=1.0,
        do_streak_suppression=False,
        do_diag_calibration=True,
        diag_calib_strength=0.20,
        init_max_iter=30,
        init_sigma=1.2,
        init_rowcol_weight=0.65,
        init_local_weight=0.35,
        init_neighbor_window=80,
        init_neighbor_tau=30.0,
        residual_smooth_window=11,
        device=args.device,
        seed=args.seed,
        verbose=True,
        debug_batch=True,
    )
    mat_final_z = result["mat_final"].astype(np.float32)
    mat_rec = restore_from_diagonal_zscore(
        mat_z=mat_final_z,
        diag_stats=diag_stats,
        max_diag=args.mask_max_diag,
        fill_diagonal=np.nan,
        keep_outside=False,
        outside_value=np.nan,
        clip_nonnegative=True,
    )
    prefix = outdir / f"{args.sample_name}_EMMA_{chrom}_{args.res_label}_mask"
    filled = high.astype(np.float32)
    high_masked = high.astype(np.float32)
    pred = np.full(high.shape, np.nan, dtype=np.float32)
    high_masked[full_mask] = 0.0
    full_pred_region = np.full((work_n, work_n), np.nan, dtype=np.float32)
    valid_work = work_mask & np.isfinite(mat_rec)
    full_pred_region[valid_work] = mat_rec[valid_work]
    pred[:work_n, :work_n] = full_pred_region
    valid_full = full_mask & np.isfinite(pred)
    filled[full_mask] = 0.0
    filled[valid_full] = pred[valid_full]
    filled = np.maximum(filled, filled.T)
    np.save(f"{prefix}_filled.npy", filled.astype(np.float32))
    np.save(f"{prefix}_prediction_only.npy", pred.astype(np.float32))
    np.save(f"{prefix}_mask.npy", full_mask)
    np.save(f"{prefix}_high_masked.npy", high_masked.astype(np.float32))
    meta = {
        "chrom": chrom,
        "shape": list(high.shape),
        "bin_limit": int(args.bin_limit) if args.bin_limit is not None else None,
        "work_n": int(work_n),
        "mask_points": int(full_mask.sum()),
        "selected_bins": selected_bins.tolist(),
        "regions": [list(r) for r in regions],
        "valid_predicted_mask_points": int(valid_full.sum()),
        "uncovered_mask_points_zeroed": int((full_mask & ~np.isfinite(pred)).sum()),
        "resolution": args.resolution,
        "resolution_label": args.res_label,
        "sample_name": args.sample_name,
        "source": "EMMA Python implementation",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_loss": float(result["reports"]["train_report"]["best_loss"]),
        "mask": {
            "block_sizes": list(args.block_sizes),
            "gap": args.gap,
            "start_bin": args.start_bin,
            "max_diag": args.mask_max_diag,
        },
    }
    Path(f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


def info(args):
    clr = cooler.Cooler(cooler_uri(args.mcool, args.resolution))
    chroms = [normalize_chrom(c) for c in args.chroms]
    print(json.dumps({
        "uri": cooler_uri(args.mcool, args.resolution),
        "binsize": int(clr.binsize),
        "chroms": {
            c: {"size_bp": int(clr.chromsizes[c]), "bins": int(math.ceil(int(clr.chromsizes[c]) / args.resolution))}
            for c in chroms
        },
    }, indent=2))


def predict_mask(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for chrom in args.chroms:
        summary.append(run_chrom(args, chrom))
    (outdir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(description="EMMA missing-region imputation for chromatin contact maps")
    p.add_argument(
        "--mcool",
        default=os.environ.get("EMMA_MCOOL", "data/H1_ESC_4DNFI82R42AD.mcool"),
    )
    p.add_argument("--resolution", type=int, default=10000)
    p.add_argument("--sample-name", default="H1_ESC")
    p.add_argument("--res-label", default="10kb")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("info")
    pi.add_argument("--chroms", nargs="+", default=["chr2", "chr3", "chr4", "chr6", "chr7"])
    pi.set_defaults(func=info)
    pp = sub.add_parser("predict-mask")
    pp.add_argument("--chroms", nargs="+", default=["chr2", "chr3", "chr4", "chr6", "chr7"])
    pp.add_argument(
        "--output-dir",
        default=os.environ.get("EMMA_OUTPUT_DIR", "results/mask_predictions"),
    )
    pp.add_argument("--bin-limit", type=int, default=2000)
    pp.add_argument("--device", default="cuda:0")
    pp.add_argument("--seed", type=int, default=2026)
    pp.add_argument("--epochs", type=int, default=20)
    pp.add_argument("--batch-size", type=int, default=128)
    pp.add_argument("--lr", type=float, default=1e-4)
    pp.add_argument("--base-channels", type=int, default=32)
    pp.add_argument("--depth", type=int, default=3)
    pp.add_argument("--patch-len", type=int, default=256)
    pp.add_argument("--n-samples", type=int, default=30000)
    pp.add_argument("--pseudo-mask-ratio", type=float, default=0.15)
    pp.add_argument("--block-sizes", nargs="+", type=int, default=[5, 10, 15, 20, 25, 30])
    pp.add_argument("--gap", type=int, default=300)
    pp.add_argument("--start-bin", type=int, default=200)
    pp.add_argument("--mask-max-diag", type=int, default=500)
    pp.set_defaults(func=predict_mask)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
