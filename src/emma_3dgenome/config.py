from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass
class EmmaConfig:
    max_diag: int = 500
    min_diag: int = 1
    max_imfs: int = 5
    imf_weights: tuple[float, ...] = (0.08, 1.35, 1.20, 1.90, 0.80)
    residual_weight: float = 1.0
    diag_calib_strength: float = 0.20

    patch_len: int = 256
    stride: int = 128
    n_samples: int = 30000
    pseudo_mask_ratio: float = 0.15
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    base_channels: int = 32
    depth: int = 3
    dropout: float = 0.05
    recompute_pseudo_emd: bool = False

    init_max_iter: int = 30
    init_sigma: float = 1.2
    init_rowcol_weight: float = 0.65
    init_local_weight: float = 0.35
    init_neighbor_window: int = 80
    init_neighbor_tau: float = 30.0
    residual_smooth_window: int = 11

    replace_strength: float = 1.0
    obs_imf_beta: float = 0.0
    inference_batch_size: int = 256
    do_diag_calibration: bool = True
    do_streak_suppression: bool = False
    streak_strength: float = 0.04

    seed: int = 2026
    device: str | None = None
    num_workers: int = 0
    prefetch_factor: int = 2
    verbose: bool = False

    def validate(self) -> None:
        if self.max_imfs <= 0:
            raise ValueError("max_imfs must be > 0.")
        if len(self.imf_weights) != self.max_imfs:
            raise ValueError(
                f"len(imf_weights) must equal max_imfs. "
                f"Got len(imf_weights)={len(self.imf_weights)}, max_imfs={self.max_imfs}."
            )
        if self.max_diag <= 0:
            raise ValueError("max_diag must be > 0.")
        if self.min_diag < 1:
            raise ValueError("min_diag must be >= 1.")
        if self.min_diag > self.max_diag:
            raise ValueError("min_diag must be <= max_diag.")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be > 0.")
        if self.patch_len <= 0:
            raise ValueError("patch_len must be > 0.")
        if self.stride <= 0:
            raise ValueError("stride must be > 0.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be > 0.")
        if not 0 <= self.pseudo_mask_ratio < 1:
            raise ValueError("pseudo_mask_ratio must satisfy 0 <= pseudo_mask_ratio < 1.")
        if self.residual_weight < 0:
            raise ValueError("residual_weight must be >= 0.")
        if self.diag_calib_strength < 0:
            raise ValueError("diag_calib_strength must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["imf_weights"] = list(self.imf_weights)
        return out


PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "paper": {},
    "smooth": {
        "imf_weights": (0.03, 0.90, 1.10, 1.30, 0.90),
        "diag_calib_strength": 0.25,
    },
    "sharp": {
        "imf_weights": (0.05, 1.40, 1.30, 2.00, 0.80),
        "diag_calib_strength": 0.15,
    },
    "conservative": {
        "imf_weights": (0.10, 1.00, 1.00, 1.10, 0.95),
        "diag_calib_strength": 0.10,
    },
    "fast": {
        "epochs": 5,
        "n_samples": 5000,
        "batch_size": 64,
        "max_diag": 300,
    },
}


def get_preset_config(preset: str = "default") -> EmmaConfig:
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {valid}.")
    config = EmmaConfig(**PRESETS[preset])
    config.validate()
    return config


def merge_config(config: EmmaConfig | None = None, preset: str = "default", **overrides: Any) -> EmmaConfig:
    base = config if config is not None else get_preset_config(preset)
    clean = {k: v for k, v in overrides.items() if v is not None}
    if "imf_weights" in clean and not isinstance(clean["imf_weights"], tuple):
        clean["imf_weights"] = tuple(float(x) for x in clean["imf_weights"])
    merged = replace(base, **clean)
    merged.validate()
    return merged


def validate_config(config: EmmaConfig) -> None:
    config.validate()
