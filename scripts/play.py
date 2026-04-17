"""CLI: load a checkpoint and watch (or export) the trained agent play 2048.

Usage:
    python scripts/play.py --checkpoint checkpoints/latest.pkl --viz
    python scripts/play.py --checkpoint checkpoints/latest.pkl --save playthrough.gif
    python scripts/play.py --checkpoint checkpoints/latest.pkl              # text only

By default, uses the deploy-time actor (NN_r + NN_p, no MCTS) per the
project spec; pass --mcts to use u-MCTS at inference.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import build_system
from src.util.config import Config


_ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def run(cfg_path: str, checkpoint: str, temperature: float, viz: bool,
        save_path: str | None, mcts: bool, interval_ms: int, seed_override: int | None):
    cfg = Config.load(cfg_path)
    if seed_override is not None:
        cfg.seed = seed_override
    sys_ = build_system(cfg)
    nnm = sys_["nnm"]
    asm = sys_["asm"]
    rlm = sys_["rlm"]

    params, enc_params, opt_state, meta = nnm.load(
        checkpoint, sys_["params"], sys_["enc_params"], sys_["opt_state"],
    )
    asm.set_params(params)
    print(f"loaded checkpoint: {checkpoint}  (trained through episode {meta.get('episode')})")

    if mcts:
        # Self-play path: uses u-MCTS + records transitions.
        episode = rlm.run_episode(temperature=temperature)
        frames = _frames_from_episode(episode)
    else:
        frames = _collect_play_actor(rlm, temperature)

    if not frames:
        print("agent had no legal move from the start — nothing to render.")
        return

    total_reward = sum(f["reward"] for f in frames)
    max_tile_log2 = max(int(f["state"].max()) for f in frames)
    max_tile = 1 << max_tile_log2 if max_tile_log2 > 0 else 0
    print(f"episode length: {len(frames)}  total reward: {total_reward:.0f}  "
          f"max tile: {max_tile}")

    if viz or save_path is not None:
        from src.viz.play_animation import animate_play
        animate_play(
            frames, interval_ms=interval_ms, save_path=save_path, show=viz,
        )
    else:
        for i, f in enumerate(frames):
            print(f"step {i:3d}  action={_ACTION_NAMES[f['action']]:>5s}  "
                  f"r={f['reward']:.0f}  cum={f['cum_reward']:.0f}")


def _collect_play_actor(rlm, temperature: float):
    frames, cum = [], 0.0
    for state, pi, action, reward, next_state in rlm.play_actor(temperature=temperature):
        cum += float(reward)
        frames.append({
            "state": state, "pi": pi, "action": int(action),
            "reward": float(reward), "cum_reward": cum, "next_state": next_state,
        })
    return frames


def _frames_from_episode(episode):
    import numpy as np
    frames, cum = [], 0.0
    pi_arr = episode.policies
    for i, action in enumerate(episode.actions):
        cum += float(episode.rewards[i])
        pi = pi_arr[i] if i < len(pi_arr) else np.zeros(pi_arr.shape[1] if pi_arr.size else 4)
        frames.append({
            "state": np.asarray(episode.states[i]),
            "pi": np.asarray(pi),
            "action": int(action),
            "reward": float(episode.rewards[i]),
            "cum_reward": cum,
            "next_state": np.asarray(episode.states[i + 1]),
        })
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature (0 = argmax)")
    p.add_argument("--viz", action="store_true", help="show the matplotlib animation")
    p.add_argument("--save", default=None, help="save animation to this path (.gif/.mp4)")
    p.add_argument("--mcts", action="store_true",
                   help="use u-MCTS at inference (default: NN_r+NN_p actor only)")
    p.add_argument("--interval_ms", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    run(args.config, args.checkpoint, args.temperature, args.viz,
        args.save, args.mcts, args.interval_ms, args.seed)


if __name__ == "__main__":
    main()
