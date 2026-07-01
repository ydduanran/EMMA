from __future__ import annotations


def masked_imf_autoencoder_loss(*args, **kwargs):
    from .engine import masked_imf_autoencoder_loss as _fn

    return _fn(*args, **kwargs)
