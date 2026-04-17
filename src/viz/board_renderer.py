"""Render a 2048 board with the canonical palette.

States are stored as log2 exponents (0 means empty), so we translate the
grid cell to its visual tile value 2**log2 for display.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Canonical 2048 colour palette (tile value -> (background, foreground)).
_PALETTE: Dict[int, Tuple[str, str]] = {
    0:     ("#cdc1b4", "#776e65"),
    2:     ("#eee4da", "#776e65"),
    4:     ("#ede0c8", "#776e65"),
    8:     ("#f2b179", "#f9f6f2"),
    16:    ("#f59563", "#f9f6f2"),
    32:    ("#f67c5f", "#f9f6f2"),
    64:    ("#f65e3b", "#f9f6f2"),
    128:   ("#edcf72", "#f9f6f2"),
    256:   ("#edcc61", "#f9f6f2"),
    512:   ("#edc850", "#f9f6f2"),
    1024:  ("#edc53f", "#f9f6f2"),
    2048:  ("#edc22e", "#f9f6f2"),
    4096:  ("#3c3a32", "#f9f6f2"),
    8192:  ("#3c3a32", "#f9f6f2"),
    16384: ("#3c3a32", "#f9f6f2"),
    32768: ("#3c3a32", "#f9f6f2"),
}


def tile_colours(value: int) -> Tuple[str, str]:
    return _PALETTE.get(value, ("#3c3a32", "#f9f6f2"))


def draw_board(ax, state: np.ndarray, *, title: str | None = None) -> None:
    """Draw a single 2048 board onto a Matplotlib axis.

    `state` is a (H, W) int array of log2 exponents (0 = empty).
    """
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    H, W = state.shape
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-0.05, H + 0.05)
    ax.invert_yaxis()

    # Board background.
    ax.add_patch(_rect(0, 0, W, H, "#bbada0"))

    pad = 0.06
    for r in range(H):
        for c in range(W):
            log2v = int(state[r, c])
            value = 0 if log2v <= 0 else (1 << log2v)
            bg, fg = tile_colours(value)
            ax.add_patch(_rect(c + pad, r + pad, 1 - 2 * pad, 1 - 2 * pad, bg))
            if value > 0:
                ax.text(
                    c + 0.5, r + 0.5, str(value),
                    ha="center", va="center",
                    fontsize=_fontsize_for(value),
                    color=fg, fontweight="bold",
                )

    if title is not None:
        ax.set_title(title, fontsize=11)


def _rect(x, y, w, h, color):
    import matplotlib.patches as mpatches
    return mpatches.Rectangle((x, y), w, h, linewidth=0, facecolor=color)


def _fontsize_for(value: int) -> int:
    if value < 100:
        return 22
    if value < 1000:
        return 18
    if value < 10000:
        return 15
    return 12
