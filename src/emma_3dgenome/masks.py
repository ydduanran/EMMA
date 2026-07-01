from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


AUTO_MASK_PRESETS = {
    "conservative": {"coverage_quantile": 0.01, "min_nonzero_ratio": 0.005, "require_row_and_col_low": True, "min_region_len": 2, "merge_gap": 1},
    "balanced": {"coverage_quantile": 0.05, "min_nonzero_ratio": 0.02, "require_row_and_col_low": False, "min_region_len": 2, "merge_gap": 1},
    "aggressive": {"coverage_quantile": 0.10, "min_nonzero_ratio": 0.05, "require_row_and_col_low": False, "min_region_len": 1, "merge_gap": 2},
}


@dataclass
class MaskInfo:
    mask: np.ndarray
    missing_bins: list[int]
    regions: list[tuple[int, int]]
    diagnostics: Any | None = None
    excluded_bins: list[int] | None = None

    def save(
        self,
        output_dir: str | Path,
        chrom: str | None = None,
        resolution: int | None = None,
        bin_offset: int = 0,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "mask.npy", self.mask.astype(bool))
        _dataframe({"bin": self.missing_bins}).to_csv(output / "detected_missing_bins.tsv", sep="\t", index=False)
        if chrom and resolution:
            (output / "detected_missing_regions.bed").write_text(
                regions_to_bed(self.regions, chrom, resolution, bin_offset=bin_offset),
                encoding="utf-8",
            )
        else:
            _dataframe(self.regions, columns=["start_bin", "end_bin"]).to_csv(output / "detected_missing_regions.bed", sep="\t", index=False, header=False)
        if self.excluded_bins is not None:
            _dataframe({"bin": self.excluded_bins}).to_csv(output / "excluded_bins.tsv", sep="\t", index=False)
        if self.diagnostics is not None:
            self.diagnostics.to_csv(output / "auto_mask_diagnostics.tsv", sep="\t", index=False)


class _SimpleDataFrame:
    def __init__(self, data: Any, columns: list[str] | None = None):
        if isinstance(data, dict):
            self.columns = list(data.keys()) if columns is None else columns
            lengths = []
            for value in data.values():
                if isinstance(value, (list, tuple, np.ndarray)):
                    lengths.append(len(value))
            n_rows = max(lengths) if lengths else 1
            self.rows = []
            for i in range(n_rows):
                row = []
                for col in self.columns:
                    value = data[col]
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    if isinstance(value, (list, tuple)):
                        row.append(value[i])
                    else:
                        row.append(value)
                self.rows.append(row)
        else:
            self.columns = columns or []
            self.rows = [list(row) for row in data]

    def to_csv(self, path: str | Path, sep: str = "\t", index: bool = False, header: bool = True) -> None:
        del index
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if header and self.columns:
            lines.append(sep.join(str(x) for x in self.columns))
        for row in self.rows:
            lines.append(sep.join(str(x) for x in row))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _get_pandas():
    try:
        import pandas as pd

        return pd
    except Exception:
        return None


def _dataframe(data: Any, columns: list[str] | None = None):
    pd = _get_pandas()
    if pd is not None:
        return pd.DataFrame(data, columns=columns)
    return _SimpleDataFrame(data, columns=columns)


def mask_from_bins(n_bins: int, bins: list[int] | np.ndarray, max_diag: int | None = None) -> np.ndarray:
    mask = np.zeros((int(n_bins), int(n_bins)), dtype=bool)
    bins_arr = np.array(sorted(set(int(b) for b in bins if 0 <= int(b) < n_bins)), dtype=int)
    if bins_arr.size == 0:
        return mask
    if max_diag is None:
        mask[bins_arr, :] = True
        mask[:, bins_arr] = True
    else:
        d = int(max_diag)
        for b in bins_arr:
            lo = max(0, int(b) - d)
            hi = min(int(n_bins), int(b) + d + 1)
            mask[int(b), lo:hi] = True
            mask[lo:hi, int(b)] = True
    return mask


def mask_from_bin_regions(n_bins: int, regions: list[tuple[int, int]], max_diag: int | None = None) -> np.ndarray:
    bins: list[int] = []
    for start, end in regions:
        bins.extend(range(max(0, int(start)), min(int(n_bins), int(end))))
    return mask_from_bins(n_bins, bins, max_diag=max_diag)


def load_mask_matrix(
    path: str | Path,
    n_bins: int | None = None,
    start_bin: int | None = None,
    end_bin: int | None = None,
) -> np.ndarray:
    mask = np.load(path).astype(bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"Mask must be a square boolean matrix. Got shape={mask.shape}.")
    if (start_bin is not None or end_bin is not None) and n_bins is not None and mask.shape != (n_bins, n_bins):
        start = 0 if start_bin is None else int(start_bin)
        end = mask.shape[0] if end_bin is None else int(end_bin)
        if start < 0 or end <= start or end > mask.shape[0]:
            raise ValueError(f"Invalid mask bin window: start_bin={start}, end_bin={end}.")
        mask = mask[start:end, start:end]
    if n_bins is not None and mask.shape != (n_bins, n_bins):
        raise ValueError(f"Mask shape must match matrix shape. Expected {(n_bins, n_bins)}, got {mask.shape}.")
    return mask


def merge_bins_to_regions(bins: list[int] | np.ndarray, min_region_len: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    sorted_bins = sorted(set(int(b) for b in bins))
    if not sorted_bins:
        return []
    regions: list[tuple[int, int]] = []
    start = prev = sorted_bins[0]
    for b in sorted_bins[1:]:
        if b <= prev + int(merge_gap) + 1:
            prev = b
            continue
        if prev + 1 - start >= int(min_region_len):
            regions.append((start, prev + 1))
        start = prev = b
    if prev + 1 - start >= int(min_region_len):
        regions.append((start, prev + 1))
    return regions


def regions_to_bed(
    regions: list[tuple[int, int]],
    chrom: str,
    resolution: int,
    bin_offset: int = 0,
) -> str:
    lines = []
    for start, end in regions:
        start_abs = int(start) + int(bin_offset)
        end_abs = int(end) + int(bin_offset)
        lines.append(f"{chrom}\t{start_abs * int(resolution)}\t{end_abs * int(resolution)}")
    return "\n".join(lines) + ("\n" if lines else "")


def _read_regions(path: str | Path, chrom: str | None = None) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split()
            if len(parts) < 3:
                continue
            c, start, end = parts[0], int(float(parts[1])), int(float(parts[2]))
            if chrom is None or c == chrom:
                rows.append((c, start, end))
    return rows


def load_mask_regions(
    path: str | Path,
    chrom: str,
    resolution: int,
    n_bins: int,
    coordinate: str = "auto",
    max_diag: int | None = None,
    bin_offset: int = 0,
) -> MaskInfo:
    raw = _read_regions(path, chrom=chrom)
    regions: list[tuple[int, int]] = []
    for _, start, end in raw:
        if coordinate == "bin":
            if bin_offset and max(start, end) > n_bins:
                s, e = start - int(bin_offset), end - int(bin_offset)
            else:
                s, e = start, end
        elif coordinate == "auto" and max(start, end) <= n_bins:
            s, e = start, end
        elif coordinate == "auto" and bin_offset and start >= int(bin_offset) and end <= int(bin_offset) + int(n_bins):
            s, e = start - int(bin_offset), end - int(bin_offset)
        else:
            s = start // int(resolution) - int(bin_offset)
            e = int(np.ceil(end / int(resolution))) - int(bin_offset)
        s = max(0, min(int(n_bins), int(s)))
        e = max(0, min(int(n_bins), int(e)))
        if e > s:
            regions.append((s, e))
    bins = [b for s, e in regions for b in range(s, e)]
    return MaskInfo(mask=mask_from_bins(n_bins, bins, max_diag=max_diag), missing_bins=sorted(set(bins)), regions=regions)


def filter_bins_by_exclude_bed(
    bins: list[int],
    chrom: str,
    resolution: int,
    exclude_bed: str | Path | None,
    bin_offset: int = 0,
) -> tuple[list[int], list[int]]:
    if exclude_bed is None:
        return bins, []
    excluded: set[int] = set()
    for _, start, end in _read_regions(exclude_bed, chrom=chrom):
        s = start // int(resolution)
        e = int(np.ceil(end / int(resolution)))
        excluded.update(range(s, e))
    kept = [b for b in bins if int(b) + int(bin_offset) not in excluded]
    removed = sorted({b for b in bins if int(b) + int(bin_offset) in excluded})
    return kept, removed


def detect_missing_bins(
    matrix: np.ndarray,
    chrom: str | None = None,
    resolution: int | None = None,
    mode: str = "balanced",
    max_diag: int = 500,
    exclude_bed: str | Path | None = None,
    min_region_len: int | None = None,
    merge_gap: int | None = None,
    bin_offset: int = 0,
) -> MaskInfo:
    if mode not in AUTO_MASK_PRESETS:
        raise ValueError(f"Unknown auto-mask mode '{mode}'.")
    preset = dict(AUTO_MASK_PRESETS[mode])
    min_region_len = preset["min_region_len"] if min_region_len is None else int(min_region_len)
    merge_gap = preset["merge_gap"] if merge_gap is None else int(merge_gap)
    mat = np.asarray(matrix, dtype=np.float32)
    n = mat.shape[0]
    finite = np.isfinite(mat)
    nonzero = finite & (mat > 0)
    row_sum = np.nan_to_num(mat, nan=0.0).sum(axis=1)
    col_sum = np.nan_to_num(mat, nan=0.0).sum(axis=0)
    row_nonzero = nonzero.mean(axis=1)
    col_nonzero = nonzero.mean(axis=0)
    finite_ratio = finite.mean(axis=1)
    coverage = 0.5 * (row_sum + col_sum)
    threshold = float(np.quantile(coverage[np.isfinite(coverage)], preset["coverage_quantile"]))
    row_low = (row_sum <= threshold) | (row_nonzero <= preset["min_nonzero_ratio"])
    col_low = (col_sum <= threshold) | (col_nonzero <= preset["min_nonzero_ratio"])
    candidate = row_low & col_low if preset["require_row_and_col_low"] else row_low | col_low
    candidate_bins = np.where(candidate)[0].astype(int).tolist()
    used_bins, excluded_bins = filter_bins_by_exclude_bed(
        candidate_bins,
        chrom or "",
        resolution or 1,
        exclude_bed,
        bin_offset=bin_offset,
    )
    regions = merge_bins_to_regions(used_bins, min_region_len=min_region_len, merge_gap=merge_gap)
    region_bins = [b for s, e in regions for b in range(s, e)]
    mask = mask_from_bins(n, region_bins, max_diag=max_diag)
    region_id = np.full(n, -1, dtype=int)
    for idx, (s, e) in enumerate(regions):
        region_id[s:e] = idx
    diagnostics = _dataframe(
        {
            "chrom": chrom,
            "bin_id": np.arange(n, dtype=int),
            "genomic_bin_id": np.arange(n, dtype=int) + int(bin_offset),
            "start": (np.arange(n, dtype=int) + int(bin_offset)) * int(resolution or 1),
            "end": (np.arange(n, dtype=int) + int(bin_offset) + 1) * int(resolution or 1),
            "row_sum": row_sum,
            "col_sum": col_sum,
            "row_nonzero_ratio": row_nonzero,
            "col_nonzero_ratio": col_nonzero,
            "finite_ratio": finite_ratio,
            "coverage_score": coverage,
            "is_candidate_missing": candidate,
            "is_excluded": np.isin(np.arange(n), excluded_bins),
            "is_used_for_imputation": np.isin(np.arange(n), region_bins),
            "region_id": region_id,
        }
    )
    return MaskInfo(mask=mask, missing_bins=region_bins, regions=regions, diagnostics=diagnostics, excluded_bins=excluded_bins)
