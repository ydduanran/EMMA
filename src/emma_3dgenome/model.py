from __future__ import annotations


def __getattr__(name: str):
    if name in {"ConvBlock1D", "MaskedIMFAutoEncoder1D", "MaskedIMFAutoEncoderDataset"}:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(name)


__all__ = ["ConvBlock1D", "MaskedIMFAutoEncoder1D", "MaskedIMFAutoEncoderDataset"]
