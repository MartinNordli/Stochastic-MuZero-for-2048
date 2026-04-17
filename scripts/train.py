"""CLI entry point: train Stochastic MuZero on 2048.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --N_e 100 --N_es 200 --M_s 10
"""
from __future__ import annotations

# Force CPU backend: JAX's Metal backend on Apple Silicon is experimental and
# has missing ops. For the network sizes in this project CPU is plenty fast.
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import episode_loop
from src.util.config import Config


def override_cfg(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides (useful for quick smoke tests)."""
    t = cfg.training
    s = cfg.search
    if args.N_e is not None:
        t.N_e = args.N_e
    if args.N_es is not None:
        t.N_es = args.N_es
    if args.M_s is not None:
        s.M_s = args.M_s
    if args.mbs is not None:
        t.mbs = args.mbs
    if args.I_t is not None:
        t.I_t = args.I_t
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--N_e", type=int, default=None, help="number of episodes")
    parser.add_argument("--N_es", type=int, default=None, help="max steps per episode")
    parser.add_argument("--M_s", type=int, default=None, help="MCTS simulations per move")
    parser.add_argument("--mbs", type=int, default=None, help="minibatch size")
    parser.add_argument("--I_t", type=int, default=None, help="train every I_t episodes")
    parser.add_argument("--train_steps_per_update", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg = override_cfg(cfg, args)
    episode_loop(cfg, train_steps_per_update=args.train_steps_per_update)


if __name__ == "__main__":
    main()
