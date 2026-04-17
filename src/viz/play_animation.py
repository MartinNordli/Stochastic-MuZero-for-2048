"""Matplotlib FuncAnimation of a trained agent playing 2048.

Consumes the output of `RLManager.play_actor(...)` — a generator yielding
`(state, pi, action, reward, next_state)` tuples per step. The figure shows
the board on the left and the actor's policy (π) as a bar chart on the
right, with cumulative reward + chosen action annotated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.viz.board_renderer import draw_board

_ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def collect_frames(play_iter: Iterable[Tuple]) -> List[dict]:
    """Materialise a play_actor generator into a list of frame dicts."""
    frames: List[dict] = []
    cum_reward = 0.0
    for state, pi, action, reward, next_state in play_iter:
        cum_reward += float(reward)
        frames.append({
            "state": np.asarray(state),
            "pi": np.asarray(pi),
            "action": int(action),
            "reward": float(reward),
            "cum_reward": cum_reward,
            "next_state": np.asarray(next_state),
        })
    return frames


def animate_play(
    frames: List[dict],
    *,
    interval_ms: int = 500,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Build (and optionally save/show) the FuncAnimation."""
    if not frames:
        raise ValueError("No frames to animate — did the game terminate immediately?")

    fig, (ax_board, ax_pi) = plt.subplots(
        1, 2, figsize=(9, 4.5), gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.suptitle("Stochastic MuZero — 2048", fontsize=13)

    def render(i: int):
        frame = frames[i]
        title = (f"step {i}  |  action: {_ACTION_NAMES[frame['action']]}  "
                 f"|  reward: {frame['reward']:.0f}  "
                 f"|  total: {frame['cum_reward']:.0f}")
        draw_board(ax_board, frame["state"], title=title)

        ax_pi.clear()
        ax_pi.set_title("policy π (from NN_r + NN_p)", fontsize=10)
        bars = ax_pi.bar(_ACTION_NAMES, frame["pi"], color="#8f7a66")
        bars[frame["action"]].set_color("#f65e3b")
        ax_pi.set_ylim(0, 1)
        ax_pi.set_ylabel("probability")
        for spine in ("top", "right"):
            ax_pi.spines[spine].set_visible(False)

    anim = FuncAnimation(
        fig, render, frames=len(frames), interval=interval_ms, repeat=False,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer="pillow", fps=max(1, 1000 // interval_ms))
        else:
            anim.save(save_path, fps=max(1, 1000 // interval_ms))

    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim
