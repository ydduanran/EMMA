from pathlib import Path

import numpy as np
import pytest

from emma_3dgenome import EmmaRestorer
from emma_3dgenome.masks import mask_from_bins


pytest.importorskip("PyEMD")
pytest.importorskip("torch")


def _small_contact_matrix(n=64):
    rng = np.random.default_rng(2)
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
    base = np.exp(-dist / 12.0)
    noise = 0.05 * rng.random((n, n), dtype=np.float32)
    mat = base + 0.5 * (noise + noise.T)
    np.fill_diagonal(mat, 1.0)
    return mat.astype(np.float32)


def test_restore_small_keeps_observed_and_saves(tmp_path: Path):
    mat = _small_contact_matrix()
    mask = mask_from_bins(mat.shape[0], [20, 21], max_diag=20)
    restorer = EmmaRestorer(
        preset="fast",
        epochs=0,
        max_diag=20,
        max_imfs=2,
        imf_weights=(1.0, 1.0),
        init_max_iter=1,
    )
    result = restorer.restore(mat, mask=mask, regions=[(20, 22)])
    assert result.restored_matrix.shape == mat.shape
    assert np.allclose(result.restored_matrix[~mask], mat[~mask], atol=1e-6)

    idx = np.arange(mat.shape[0])
    dist = np.abs(idx[:, None] - idx[None, :])
    target = mask & (dist >= 1) & (dist <= 20)
    assert np.isfinite(result.restored_matrix[target]).all()

    result.save(tmp_path)
    assert (tmp_path / "restored.npy").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "log.txt").exists()
