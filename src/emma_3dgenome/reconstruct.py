from __future__ import annotations

import numpy as np

from .config import EmmaConfig
from .restore import EmmaRestorer


def reconstruct_matrix(
    matrix: np.ndarray,
    config: EmmaConfig | None = None,
    mode: str = "conservative",
    blend: float | None = 0.2,
):
    restorer = EmmaRestorer(config=config)
    return restorer.reconstruct(matrix, mode=mode, blend=blend)
