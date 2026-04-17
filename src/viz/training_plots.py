"""Plot training curves from a JSONL training log.

The log emitted by `episode_loop` is one JSON object per line with fields
`episode`, `reward`, `max_tile`, `buffer_size`, `loss` (dict or null), ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def load_log(path: str | Path) -> List[dict]:
    records: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_training(records: List[dict], *, save_path: str | Path | None = None, show: bool = True):
    """Draw a 2x3 grid of diagnostic plots."""
    if not records:
        raise ValueError("Empty training log.")

    eps = np.array([r["episode"] for r in records])
    rewards = np.array([r["reward"] for r in records], dtype=float)
    lengths = np.array([r["length"] for r in records], dtype=float)
    max_tiles = np.array([r["max_tile"] for r in records], dtype=float)

    loss_eps, loss_total, loss_pol, loss_val, loss_rew, loss_ch = [], [], [], [], [], []
    for r in records:
        L = r.get("loss")
        if L is None:
            continue
        loss_eps.append(r["episode"])
        loss_total.append(L["total"])
        loss_pol.append(L["policy"])
        loss_val.append(L["value"])
        loss_rew.append(L["reward"])
        loss_ch.append(L["chance"])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Stochastic MuZero — training diagnostics", fontsize=13)

    ax = axes[0, 0]
    ax.plot(eps, rewards, color="#4a90d9")
    ax.plot(eps, _smooth(rewards, 20), color="#1a3e6e", lw=1.5, label="smoothed")
    ax.set_title("episode reward")
    ax.set_xlabel("episode")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    ax.plot(eps, lengths, color="#6aa84f")
    ax.plot(eps, _smooth(lengths, 20), color="#274e13", lw=1.5)
    ax.set_title("episode length")
    ax.set_xlabel("episode")

    ax = axes[0, 2]
    ax.plot(eps, max_tiles, color="#cc4125")
    ax.set_yscale("log", base=2)
    ax.set_title("max tile reached")
    ax.set_xlabel("episode")

    ax = axes[1, 0]
    if loss_eps:
        ax.plot(loss_eps, loss_total, color="#333")
    ax.set_title("total loss")
    ax.set_xlabel("episode")

    ax = axes[1, 1]
    if loss_eps:
        ax.plot(loss_eps, loss_pol, label="policy", color="#4a90d9")
        ax.plot(loss_eps, loss_val, label="value", color="#cc4125")
        ax.plot(loss_eps, loss_rew, label="reward", color="#6aa84f")
        ax.plot(loss_eps, loss_ch, label="chance", color="#a64d79")
        ax.legend(fontsize=8)
    ax.set_title("loss components")
    ax.set_xlabel("episode")

    ax = axes[1, 2]
    ax.hist(max_tiles, bins=np.unique(max_tiles), color="#cc4125", edgecolor="white")
    ax.set_xscale("log", base=2)
    ax.set_title("max-tile histogram")
    ax.set_xlabel("tile")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < 2 or window <= 1:
        return x
    w = min(window, len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")
