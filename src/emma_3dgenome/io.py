from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import symmetrize
from .utils import ensure_square_matrix, normalize_chrom


def normalize_bin_window(
    n_bins: int,
    start_bin: int | None = None,
    end_bin: int | None = None,
    bin_limit: int | None = None,
) -> tuple[int, int]:
    if start_bin is not None and int(start_bin) < 0:
        raise ValueError("--start-bin must be >= 0.")
    if end_bin is not None and int(end_bin) < 0:
        raise ValueError("--end-bin must be >= 0.")
    if bin_limit is not None and end_bin is not None:
        raise ValueError("Use either --end-bin or --bin-limit, not both.")

    start = 0 if start_bin is None else int(start_bin)
    if start > int(n_bins):
        raise ValueError(f"--start-bin={start} is outside the matrix/chromosome with {n_bins} bins.")

    if end_bin is not None:
        end = int(end_bin)
    elif bin_limit is not None:
        end = start + int(bin_limit)
    else:
        end = int(n_bins)

    end = min(end, int(n_bins))
    if end <= start:
        raise ValueError(f"Invalid bin window: start_bin={start}, end_bin={end}.")
    return start, end


def slice_square_matrix(matrix: np.ndarray, start_bin: int | None = None, end_bin: int | None = None) -> np.ndarray:
    mat = ensure_square_matrix(matrix)
    if start_bin is None and end_bin is None:
        return mat
    start, end = normalize_bin_window(mat.shape[0], start_bin=start_bin, end_bin=end_bin)
    return mat[start:end, start:end]


def _clean_matrix(matrix: np.ndarray, sym_mode: str = "average") -> np.ndarray:
    mat = ensure_square_matrix(matrix)
    return symmetrize(mat, mode=sym_mode)


def load_numpy_matrix(
    path: str | Path,
    key: str | None = None,
    start_bin: int | None = None,
    end_bin: int | None = None,
) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return _clean_matrix(slice_square_matrix(np.load(path), start_bin=start_bin, end_bin=end_bin))
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            key = "matrix" if "matrix" in data.files else data.files[0]
        return _clean_matrix(slice_square_matrix(data[key], start_bin=start_bin, end_bin=end_bin))
    raise ValueError(f"Unsupported numpy matrix file: {path}")


def load_cool_matrix(
    path: str | Path,
    chrom: str,
    resolution: int | None = None,
    bin_limit: int | None = None,
    balance: bool = False,
    start_bin: int | None = None,
    end_bin: int | None = None,
) -> np.ndarray:
    import cooler

    path = Path(path)
    if path.suffix == ".mcool":
        if resolution is None:
            raise ValueError(".mcool input requires --resolution.")
        uri = f"{path}::resolutions/{int(resolution)}"
    else:
        uri = str(path)
    clr = cooler.Cooler(uri)
    chrom = normalize_chrom(chrom)
    if chrom not in clr.chromnames:
        raise ValueError(f"Chromosome '{chrom}' was not found in {path}.")
    chrom_start, chrom_end = clr.extent(chrom)
    n_chrom = int(chrom_end - chrom_start)
    local_start, local_end = normalize_bin_window(
        n_chrom,
        start_bin=start_bin,
        end_bin=end_bin,
        bin_limit=bin_limit,
    )
    abs_start = int(chrom_start) + local_start
    abs_end = int(chrom_start) + local_end
    mat = clr.matrix(balance=balance)[abs_start:abs_end, abs_start:abs_end]
    return _clean_matrix(np.asarray(mat, dtype=np.float32))


def load_contact_matrix(
    path: str | Path,
    chrom: str | None = None,
    resolution: int | None = None,
    bin_limit: int | None = None,
    balance: bool = False,
    key: str | None = None,
    start_bin: int | None = None,
    end_bin: int | None = None,
) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return load_numpy_matrix(path, key=key, start_bin=start_bin, end_bin=end_bin)
    if suffix in {".cool", ".mcool"}:
        if chrom is None:
            raise ValueError("--chrom is required for .cool/.mcool input.")
        return load_cool_matrix(
            path,
            chrom=chrom,
            resolution=resolution,
            bin_limit=bin_limit,
            balance=balance,
            start_bin=start_bin,
            end_bin=end_bin,
        )
    raise ValueError(f"Unsupported input format: {path}. Supported: .cool, .mcool, .npy, .npz.")


def save_matrix(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float32))


def get_chrom_info(path: str | Path, resolution: int | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".mcool":
        if resolution is None:
            raise ValueError(".mcool input requires --resolution.")
        uri = f"{path}::resolutions/{int(resolution)}"
    elif path.suffix == ".cool":
        uri = str(path)
    else:
        matrix = load_contact_matrix(path)
        return {"path": str(path), "shape": list(matrix.shape)}
    import cooler

    clr = cooler.Cooler(uri)
    return {
        "uri": uri,
        "binsize": int(clr.binsize),
        "chroms": {c: {"size_bp": int(clr.chromsizes[c]), "bins": int(clr.extent(c)[1] - clr.extent(c)[0])} for c in clr.chromnames},
    }
