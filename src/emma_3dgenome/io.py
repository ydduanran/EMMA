from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import symmetrize
from .utils import ensure_square_matrix, normalize_chrom


def _clean_matrix(matrix: np.ndarray, sym_mode: str = "average") -> np.ndarray:
    mat = ensure_square_matrix(matrix)
    return symmetrize(mat, mode=sym_mode)


def load_numpy_matrix(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return _clean_matrix(np.load(path))
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            key = "matrix" if "matrix" in data.files else data.files[0]
        return _clean_matrix(data[key])
    raise ValueError(f"Unsupported numpy matrix file: {path}")


def load_cool_matrix(
    path: str | Path,
    chrom: str,
    resolution: int | None = None,
    bin_limit: int | None = None,
    balance: bool = False,
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
    start, end = clr.extent(chrom)
    n = int(end - start) if bin_limit is None else min(int(bin_limit), int(end - start))
    mat = clr.matrix(balance=balance)[start : start + n, start : start + n]
    return _clean_matrix(np.asarray(mat, dtype=np.float32))


def load_contact_matrix(
    path: str | Path,
    chrom: str | None = None,
    resolution: int | None = None,
    bin_limit: int | None = None,
    balance: bool = False,
    key: str | None = None,
) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return load_numpy_matrix(path, key=key)
    if suffix in {".cool", ".mcool"}:
        if chrom is None:
            raise ValueError("--chrom is required for .cool/.mcool input.")
        return load_cool_matrix(path, chrom=chrom, resolution=resolution, bin_limit=bin_limit, balance=balance)
    raise ValueError(f"Unsupported input format: {path}. Supported: .cool, .mcool, .npy, .npz.")


def save_matrix(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(matrix, dtype=np.float32))


def get_chrom_info(path: str | Path, resolution: int | None = None) -> dict[str, Any]:
    import cooler

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
    clr = cooler.Cooler(uri)
    return {
        "uri": uri,
        "binsize": int(clr.binsize),
        "chroms": {c: {"size_bp": int(clr.chromsizes[c]), "bins": int(clr.extent(c)[1] - clr.extent(c)[0])} for c in clr.chromnames},
    }

