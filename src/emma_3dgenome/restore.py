from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from .config import EmmaConfig, merge_config
from .io import load_contact_matrix
from .masks import MaskInfo, detect_missing_bins, load_mask_matrix, load_mask_regions, merge_bins_to_regions
from .preprocessing import clip_nonnegative, distance_zscore_denormalize, distance_zscore_normalize, symmetrize
from .result import EmmaResult
from .utils import ensure_square_matrix, get_device, get_peak_gpu_memory, set_seed


NO_MASK_MESSAGE = (
    "No imputation mask was provided. Use --mask, --mask-regions, or --auto-mask.\n"
    "If you want to reconstruct the matrix without explicit imputation, use `emma reconstruct`."
)


def _selected_bins_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_arr = np.asarray(mask, dtype=bool)
    return np.where(mask_arr.any(axis=0) | mask_arr.any(axis=1))[0].astype(np.int64)


def _jsonable_diag_stats(diag_stats: dict) -> dict[str, dict[str, float | int]]:
    return {str(k): {kk: (int(vv) if kk == "valid_count" else float(vv)) for kk, vv in v.items()} for k, v in diag_stats.items()}


def _run_captured(verbose: bool, fn):
    if verbose:
        return fn(), ""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = fn()
    return result, buffer.getvalue()


def _environment_report(config: EmmaConfig, runtime_seconds: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "seed": int(config.seed),
        "device": get_device(config.device),
        "runtime_seconds": float(runtime_seconds),
        "peak_gpu_memory_gb": get_peak_gpu_memory(),
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        report["torch_version"] = None
        report["cuda_available"] = False
    return report


class EmmaRestorer:
    def __init__(
        self,
        config: EmmaConfig | None = None,
        preset: str = "default",
        **overrides: Any,
    ):
        self.config = merge_config(config=config, preset=preset, **overrides)
        self.config.device = get_device(self.config.device)
        self.config.validate()
        set_seed(self.config.seed)

    def _run_fast_restore(
        self,
        mat_z: np.ndarray,
        mat_z_missing: np.ndarray,
        mask: np.ndarray,
        selected_bins: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], str]:
        cfg = self.config

        def _work():
            from . import engine

            mat_init = engine.build_rowcol_context_init_matrix(
                mat_z_missing=mat_z_missing,
                mask_block=mask,
                selected_bins=selected_bins,
                max_iter=cfg.init_max_iter,
                sigma=cfg.init_sigma,
                rowcol_weight=cfg.init_rowcol_weight,
                local_weight=cfg.init_local_weight,
                neighbor_window=cfg.init_neighbor_window,
                neighbor_tau=cfg.init_neighbor_tau,
                keep_observed=True,
                verbose=cfg.verbose,
            )
            diag_features = engine.build_all_diagonal_imf_features(
                mat_z_missing=mat_z_missing,
                mask_block=mask,
                mat_init=mat_init,
                max_diag=cfg.max_diag,
                min_diag=cfg.min_diag,
                max_imfs=cfg.max_imfs,
                residual_smooth_window=cfg.residual_smooth_window,
                verbose=cfg.verbose,
            )
            mat_emd_prior = engine.build_matrix_from_reconstructed_diagonals(
                diag_features=diag_features,
                n=mat_z.shape[0],
                max_diag=cfg.max_diag,
                min_diag=cfg.min_diag,
                use_key="imfs",
                residual_key="residual_smooth",
                imf_weights=cfg.imf_weights,
                residual_weight=cfg.residual_weight,
                residual_smooth_window=cfg.residual_smooth_window,
            )
            mat_final = mat_z_missing.copy()
            fill = mask & np.isfinite(mat_emd_prior)
            mat_final[fill] = mat_emd_prior[fill]
            observed = np.isfinite(mat_z_missing) & (~mask)
            mat_final[observed] = mat_z_missing[observed]
            mat_final = 0.5 * (mat_final + mat_final.T)
            np.fill_diagonal(mat_final, np.nan)
            if cfg.do_diag_calibration:
                mat_final = engine.diagonal_distribution_calibration(
                    mat_cur=mat_final,
                    mat_missing=mat_z_missing,
                    mask=mask,
                    max_diag=cfg.max_diag,
                    min_diag=cfg.min_diag,
                    strength=cfg.diag_calib_strength,
                )
            return mat_final.astype(np.float32), {
                "train_report": {
                    "epochs_run": 0,
                    "best_loss": None,
                    "device": cfg.device,
                    "note": "epochs=0; used context initialization plus EMD mode-weighted reconstruction.",
                }
            }

        (mat_final_z, report), log_text = _run_captured(cfg.verbose, _work)
        return mat_final_z, report, log_text

    def _run_full_restore(
        self,
        mat_z: np.ndarray,
        mat_z_missing: np.ndarray,
        mask: np.ndarray,
        selected_bins: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], str]:
        cfg = self.config

        def _work():
            from .engine import run_context_init_masked_imf_autoencoder_emd_hic_imputation

            return run_context_init_masked_imf_autoencoder_emd_hic_imputation(
                mat_z=mat_z,
                mat_z_missing=mat_z_missing,
                mask_block=mask,
                selected_bins=selected_bins,
                max_diag=cfg.max_diag,
                min_diag=cfg.min_diag,
                max_imfs=cfg.max_imfs,
                init_max_iter=cfg.init_max_iter,
                init_sigma=cfg.init_sigma,
                init_rowcol_weight=cfg.init_rowcol_weight,
                init_local_weight=cfg.init_local_weight,
                init_neighbor_window=cfg.init_neighbor_window,
                init_neighbor_tau=cfg.init_neighbor_tau,
                residual_smooth_window=cfg.residual_smooth_window,
                patch_len=cfg.patch_len,
                stride=cfg.stride,
                n_samples=cfg.n_samples,
                pseudo_mask_ratio=cfg.pseudo_mask_ratio,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                base_channels=cfg.base_channels,
                depth=cfg.depth,
                dropout=cfg.dropout,
                num_workers=cfg.num_workers,
                prefetch_factor=cfg.prefetch_factor,
                recompute_pseudo_emd=cfg.recompute_pseudo_emd,
                replace_strength=cfg.replace_strength,
                obs_imf_beta=cfg.obs_imf_beta,
                inference_batch_size=cfg.inference_batch_size,
                imf_weights=cfg.imf_weights,
                residual_weight=cfg.residual_weight,
                do_streak_suppression=cfg.do_streak_suppression,
                streak_strength=cfg.streak_strength,
                do_diag_calibration=cfg.do_diag_calibration,
                diag_calib_strength=cfg.diag_calib_strength,
                seed=cfg.seed,
                device=cfg.device,
                verbose=cfg.verbose,
                debug_batch=False,
            )

        result, log_text = _run_captured(cfg.verbose, _work)
        return result["mat_final"].astype(np.float32), result.get("reports", {}), log_text

    def restore(self, matrix: np.ndarray, mask: np.ndarray, regions: list | None = None) -> EmmaResult:
        cfg = self.config
        start = time.perf_counter()
        mat = ensure_square_matrix(matrix)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != mat.shape:
            raise ValueError(f"Mask shape must match matrix shape. Got {mask_arr.shape} and {mat.shape}.")
        selected_bins = _selected_bins_from_mask(mask_arr)
        if selected_bins.size == 0 or not np.any(mask_arr):
            raise ValueError("No missing bins detected. Nothing to restore.")

        mat_z, diag_stats = distance_zscore_normalize(mat, max_diag=cfg.max_diag)
        mat_z_missing = mat_z.copy()
        mat_z_missing[mask_arr] = np.nan

        if cfg.epochs == 0:
            mat_final_z, engine_report, log_text = self._run_fast_restore(mat_z, mat_z_missing, mask_arr, selected_bins)
        else:
            mat_final_z, engine_report, log_text = self._run_full_restore(mat_z, mat_z_missing, mask_arr, selected_bins)

        raw_pred = distance_zscore_denormalize(mat_final_z, diag_stats, max_diag=cfg.max_diag)
        restored = mat.copy()
        replace = mask_arr & np.isfinite(raw_pred)
        restored[replace] = raw_pred[replace]
        restored = clip_nonnegative(symmetrize(restored))

        masked_input = mat.copy()
        masked_input[mask_arr] = 0.0
        prediction_only = np.full_like(mat, np.nan, dtype=np.float32)
        prediction_only[replace] = restored[replace]

        runtime = time.perf_counter() - start
        report = _environment_report(cfg, runtime)
        report.update(
            {
                "mode": "restore",
                "matrix_shape": list(mat.shape),
                "mask_points": int(mask_arr.sum()),
                "selected_bins": int(selected_bins.size),
                "replaced_points": int(replace.sum()),
                "engine": engine_report,
                "log": log_text,
            }
        )
        return EmmaResult(
            restored_matrix=restored.astype(np.float32),
            prediction_only=prediction_only.astype(np.float32),
            masked_matrix=masked_input.astype(np.float32),
            mask=mask_arr,
            regions=regions,
            config=cfg.to_dict(),
            report=report,
            diag_stats=_jsonable_diag_stats(diag_stats),
            mode="restore",
        )

    def restore_auto(
        self,
        matrix: np.ndarray,
        chrom: str | None = None,
        resolution: int | None = None,
        exclude_bed: str | Path | None = None,
        auto_mask_mode: str = "balanced",
    ) -> EmmaResult:
        mask_info = detect_missing_bins(
            matrix,
            chrom=chrom,
            resolution=resolution,
            mode=auto_mask_mode,
            max_diag=self.config.max_diag,
            exclude_bed=exclude_bed,
        )
        if not mask_info.missing_bins:
            raise ValueError(
                f"No missing bins were detected under mode='{auto_mask_mode}'. "
                "Try --auto-mask-mode aggressive or provide --mask-regions manually."
            )
        result = self.restore(matrix, mask=mask_info.mask, regions=mask_info.regions)
        if result.report is not None:
            result.report["auto_mask_mode"] = auto_mask_mode
            result.report["auto_mask_detected_bins"] = int(len(mask_info.missing_bins))
            result.report["auto_mask_excluded_bins"] = int(len(mask_info.excluded_bins or []))
        return result

    def restore_from_file(
        self,
        path: str | Path,
        chrom: str | None = None,
        resolution: int | None = None,
        mask: str | Path | None = None,
        mask_regions: str | Path | None = None,
        mask_region_format: str = "auto",
        auto_mask: bool = False,
        exclude_bed: str | Path | None = None,
        auto_mask_mode: str = "balanced",
        output_dir: str | Path | None = None,
        balance: bool = False,
        key: str | None = None,
        start_bin: int | None = None,
        end_bin: int | None = None,
    ) -> EmmaResult:
        matrix = load_contact_matrix(
            path,
            chrom=chrom,
            resolution=resolution,
            balance=balance,
            key=key,
            start_bin=start_bin,
            end_bin=end_bin,
        )
        n_bins = matrix.shape[0]
        bin_offset = 0 if start_bin is None else int(start_bin)
        mask_info: MaskInfo | None = None

        if mask is not None:
            mask_arr = load_mask_matrix(mask, n_bins=n_bins, start_bin=start_bin, end_bin=end_bin)
            bins = _selected_bins_from_mask(mask_arr).astype(int).tolist()
            mask_info = MaskInfo(mask=mask_arr, missing_bins=bins, regions=merge_bins_to_regions(bins))
        elif mask_regions is not None:
            if chrom is None or resolution is None:
                raise ValueError("--chrom and --resolution are required when using --mask-regions.")
            mask_info = load_mask_regions(
                mask_regions,
                chrom=chrom,
                resolution=resolution,
                n_bins=n_bins,
                coordinate=mask_region_format,
                max_diag=self.config.max_diag,
                bin_offset=bin_offset,
            )
        elif auto_mask:
            mask_info = detect_missing_bins(
                matrix,
                chrom=chrom,
                resolution=resolution,
                mode=auto_mask_mode,
                max_diag=self.config.max_diag,
                exclude_bed=exclude_bed,
                bin_offset=bin_offset,
            )
            if not mask_info.missing_bins:
                raise ValueError(
                    f"No missing bins were detected under mode='{auto_mask_mode}'. "
                    "Try --auto-mask-mode aggressive or provide --mask-regions manually."
                )
        else:
            raise ValueError(NO_MASK_MESSAGE)

        result = self.restore(matrix, mask=mask_info.mask, regions=mask_info.regions)
        if result.report is not None:
            result.report.update(
                {
                    "input_path": str(path),
                    "chrom": chrom,
                    "resolution": resolution,
                    "start_bin": start_bin,
                    "end_bin": end_bin,
                    "bin_offset": bin_offset,
                    "mask_source": "mask" if mask is not None else "mask_regions" if mask_regions is not None else "auto_mask",
                }
            )
        if output_dir is not None:
            result.save(output_dir)
            if auto_mask or mask_regions is not None:
                mask_info.save(output_dir, chrom=chrom, resolution=resolution, bin_offset=bin_offset)
        return result

    def reconstruct(self, matrix: np.ndarray, mode: str = "conservative", blend: float | None = None) -> EmmaResult:
        cfg = self.config
        if mode not in {"conservative", "full"}:
            raise ValueError("mode must be 'conservative' or 'full'.")
        if blend is None:
            blend = 0.2 if mode == "conservative" else 1.0
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must satisfy 0 <= blend <= 1.")

        start = time.perf_counter()
        mat = ensure_square_matrix(matrix)
        mat_z, diag_stats = distance_zscore_normalize(mat, max_diag=cfg.max_diag)
        mask = np.zeros_like(mat, dtype=bool)

        def _work():
            from . import engine

            diag_features = engine.build_all_diagonal_imf_features(
                mat_z_missing=mat_z,
                mask_block=mask,
                mat_init=mat_z,
                max_diag=cfg.max_diag,
                min_diag=cfg.min_diag,
                max_imfs=cfg.max_imfs,
                residual_smooth_window=cfg.residual_smooth_window,
                verbose=cfg.verbose,
            )
            return engine.build_matrix_from_reconstructed_diagonals(
                diag_features=diag_features,
                n=mat.shape[0],
                max_diag=cfg.max_diag,
                min_diag=cfg.min_diag,
                use_key="imfs",
                residual_key="residual_smooth",
                imf_weights=cfg.imf_weights,
                residual_weight=cfg.residual_weight,
                residual_smooth_window=cfg.residual_smooth_window,
            )

        mat_rec_z, log_text = _run_captured(cfg.verbose, _work)
        raw_rec = distance_zscore_denormalize(mat_rec_z, diag_stats, max_diag=cfg.max_diag)
        finite = np.isfinite(raw_rec)
        reconstructed = mat.copy()
        reconstructed[finite] = (1.0 - blend) * mat[finite] + blend * raw_rec[finite]
        reconstructed = clip_nonnegative(symmetrize(reconstructed))

        runtime = time.perf_counter() - start
        report = _environment_report(cfg, runtime)
        report.update(
            {
                "mode": "reconstruct",
                "reconstruct_mode": mode,
                "blend": blend,
                "matrix_shape": list(mat.shape),
                "reconstructed_points": int(finite.sum()),
                "log": log_text,
            }
        )
        return EmmaResult(
            restored_matrix=reconstructed.astype(np.float32),
            prediction_only=None,
            masked_matrix=mat.astype(np.float32),
            mask=None,
            regions=None,
            config=cfg.to_dict(),
            report=report,
            diag_stats=_jsonable_diag_stats(diag_stats),
            mode="reconstruct",
        )
