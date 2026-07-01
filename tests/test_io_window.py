import numpy as np

from emma_3dgenome.io import load_contact_matrix, normalize_bin_window, slice_square_matrix


def test_normalize_bin_window_with_start_end():
    assert normalize_bin_window(100, start_bin=10, end_bin=25) == (10, 25)


def test_slice_square_matrix():
    mat = np.arange(100, dtype=np.float32).reshape(10, 10)
    sliced = slice_square_matrix(mat, start_bin=2, end_bin=5)
    assert sliced.shape == (3, 3)
    assert np.array_equal(sliced, mat[2:5, 2:5])


def test_load_contact_matrix_npy_window(tmp_path):
    mat = np.arange(100, dtype=np.float32).reshape(10, 10)
    mat = 0.5 * (mat + mat.T)
    path = tmp_path / "matrix.npy"
    np.save(path, mat)
    loaded = load_contact_matrix(path, start_bin=3, end_bin=7)
    assert loaded.shape == (4, 4)
    assert np.array_equal(loaded, mat[3:7, 3:7])
