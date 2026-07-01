from __future__ import annotations

from pathlib import Path

import numpy as np


def save_heatmap(matrix: np.ndarray, path: str | Path, cmap: str = "Reds") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("save_heatmap requires matplotlib.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.asarray(matrix), cmap=cmap, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=200)
    plt.close(fig)
