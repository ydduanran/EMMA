from __future__ import annotations

from .config import EmmaConfig, get_preset_config


def build_rowcol_context_init_matrix(mat_z_missing, mask, selected_bins=None, config: EmmaConfig | None = None):
    from .engine import build_rowcol_context_init_matrix as _build

    cfg = config or get_preset_config()
    return _build(
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
