"""CLI: plot training curves from a JSONL log file.

Usage:
    python scripts/visualize.py --log logs/train_log.jsonl
    python scripts/visualize.py --log logs/train_log.jsonl --save logs/curves.png --no-show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.training_plots import load_log, plot_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/train_log.jsonl")
    p.add_argument("--save", default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    records = load_log(args.log)
    print(f"loaded {len(records)} records from {args.log}")
    plot_training(records, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
