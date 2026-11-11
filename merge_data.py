#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_data.py - Merge raw_pairs.txt into train.csv and val.csv

Usage:
    python merge_data.py --raw data/raw_pairs.txt --train data/train.csv --val data/val.csv --val-ratio 0.2
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge raw pairs into train/val split")
    parser.add_argument("--raw", default="data/raw_pairs.txt", help="Input raw pairs file")
    parser.add_argument("--train", default="data/train.csv", help="Output train file")
    parser.add_argument("--val", default="data/val.csv", help="Output validation file")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run collect_pairs.py first.")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and "\t" in line]

    print(f"Found {len(lines)} valid pairs in {raw_path}")

    random.seed(42)
    random.shuffle(lines)

    n_val = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    Path(args.train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(args.val, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    print(f"\nTrain: {len(train_lines)} pairs -> {args.train}")
    print(f"Val: {len(val_lines)} pairs -> {args.val}")


if __name__ == "__main__":
    main()
