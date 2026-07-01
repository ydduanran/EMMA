import numpy as np

from emma_3dgenome.preprocessing import distance_zscore_denormalize, distance_zscore_normalize


def test_distance_zscore_roundtrip_on_valid_diagonals():
    rng = np.random.default_rng(1)
    mat = rng.random((48, 48), dtype=np.float32)
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)

    mat_z, stats = distance_zscore_normalize(mat, max_diag=20, min_valid=10)
    restored = distance_zscore_denormalize(mat_z, stats, max_diag=20, fill_diagonal=0.0)

    idx = np.arange(mat.shape[0])
    dist = np.abs(idx[:, None] - idx[None, :])
    valid = (dist >= 1) & (dist <= 20)
    assert np.isfinite(mat_z[valid]).all()
    assert np.allclose(restored[valid], mat[valid], atol=1e-5)
