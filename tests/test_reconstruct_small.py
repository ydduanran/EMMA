from pathlib import Path

import numpy as np
import pytest

from emma_3dgenome import EmmaRestorer


pytest.importorskip("PyEMD")
pytest.importorskip("torch")


def test_reconstruct_small_saves(tmp_path: Path):
    rng = np.random.default_rng(3)
    mat = rng.random((64, 64), dtype=np.float32)
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 1.0)

    restorer = EmmaRestorer(preset="fast", epochs=0, max_diag=20, max_imfs=2, imf_weights=(1.0, 1.0))
    result = restorer.reconstruct(mat, mode="conservative", blend=0.2)
    assert result.restored_matrix.shape == mat.shape
    assert np.isfinite(result.restored_matrix).all()

    result.save(tmp_path)
    assert (tmp_path / "reconstructed.npy").exists()
    assert (tmp_path / "difference.npy").exists()
