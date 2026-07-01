from __future__ import annotations


def init_1d_signal_for_emd(*args, **kwargs):
    from .engine import init_1d_signal_for_emd as _fn

    return _fn(*args, **kwargs)


def smooth_1d_nan_aware(*args, **kwargs):
    from .engine import smooth_1d_nan_aware as _fn

    return _fn(*args, **kwargs)


def emd_decompose_1d_fixed(*args, **kwargs):
    from .engine import emd_decompose_1d_fixed as _fn

    return _fn(*args, **kwargs)


def decompose_one_diagonal_to_imf_features(*args, **kwargs):
    from .engine import decompose_one_diagonal_to_imf_features as _fn

    return _fn(*args, **kwargs)


def build_all_diagonal_imf_features(*args, **kwargs):
    from .engine import build_all_diagonal_imf_features as _fn

    return _fn(*args, **kwargs)


def reconstruct_signal_from_imfs(*args, **kwargs):
    from .engine import reconstruct_signal_from_imfs as _fn

    return _fn(*args, **kwargs)


def build_matrix_from_reconstructed_diagonals(*args, **kwargs):
    from .engine import build_matrix_from_reconstructed_diagonals as _fn

    return _fn(*args, **kwargs)
