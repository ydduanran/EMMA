from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class EmmaResult:
    restored_matrix: np.ndarray
    prediction_only: np.ndarray | None = None
    masked_matrix: np.ndarray | None = None
    mask: np.ndarray | None = None
    regions: list | None = None
    config: dict | None = None
    report: dict | None = None
    diag_stats: dict | None = None
    mode: str = "restore"

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.mode == "reconstruct":
            np.save(output / "reconstructed.npy", self.restored_matrix.astype(np.float32))
            if self.masked_matrix is not None:
                np.save(output / "difference.npy", (self.restored_matrix - self.masked_matrix).astype(np.float32))
        else:
            np.save(output / "restored.npy", self.restored_matrix.astype(np.float32))
            if self.prediction_only is not None:
                np.save(output / "prediction_only.npy", self.prediction_only.astype(np.float32))
            if self.masked_matrix is not None:
                np.save(output / "masked_input.npy", self.masked_matrix.astype(np.float32))
            if self.mask is not None:
                np.save(output / "mask.npy", self.mask.astype(bool))
        (output / "config.json").write_text(json.dumps(self.config or {}, indent=2, default=_json_default), encoding="utf-8")
        (output / "report.json").write_text(json.dumps(self.report or {}, indent=2, default=_json_default), encoding="utf-8")
        log_text = ""
        if self.report is not None:
            log_text = str(self.report.get("log", ""))
        if not log_text:
            log_text = f"mode={self.mode}\n"
        (output / "log.txt").write_text(log_text, encoding="utf-8")
        if self.diag_stats is not None:
            (output / "diag_stats.json").write_text(json.dumps(self.diag_stats, indent=2, default=_json_default), encoding="utf-8")
        if self.regions is not None:
            with open(output / "mask_regions.bed", "w", encoding="utf-8") as f:
                for region in self.regions:
                    f.write("\t".join(str(x) for x in region) + "\n")

    @classmethod
    def load(cls, output_dir: str | Path) -> "EmmaResult":
        output = Path(output_dir)
        restored_path = output / "restored.npy"
        mode = "restore"
        if not restored_path.exists():
            restored_path = output / "reconstructed.npy"
            mode = "reconstruct"
        restored = np.load(restored_path)
        prediction = np.load(output / "prediction_only.npy") if (output / "prediction_only.npy").exists() else None
        masked = np.load(output / "masked_input.npy") if (output / "masked_input.npy").exists() else None
        mask = np.load(output / "mask.npy") if (output / "mask.npy").exists() else None
        config = json.loads((output / "config.json").read_text()) if (output / "config.json").exists() else {}
        report = json.loads((output / "report.json").read_text()) if (output / "report.json").exists() else {}
        diag_stats = json.loads((output / "diag_stats.json").read_text()) if (output / "diag_stats.json").exists() else None
        return cls(restored, prediction, masked, mask, None, config, report, diag_stats, mode=mode)
