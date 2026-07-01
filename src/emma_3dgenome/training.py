from __future__ import annotations

from .config import EmmaConfig, get_preset_config


def train_masked_imf_autoencoder(diag_features, config: EmmaConfig | None = None):
    from .engine import train_masked_imf_autoencoder as _train

    cfg = config or get_preset_config()
    return _train(
        diag_features=diag_features,
        max_diag=cfg.max_diag,
        max_imfs=cfg.max_imfs,
        residual_smooth_window=cfg.residual_smooth_window,
        patch_len=cfg.patch_len,
        n_samples=cfg.n_samples,
        pseudo_mask_ratio=cfg.pseudo_mask_ratio,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
        imf_weights=cfg.imf_weights,
        seed=cfg.seed,
        device=cfg.device,
        num_workers=cfg.num_workers,
        verbose=cfg.verbose,
        debug_batch=cfg.verbose,
    )
