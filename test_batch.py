#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch.py - run the model over a held-out set and report real examples.

Usage:
    python3 test_batch.py          # 20 examples from data/val_real.tsv
    python3 test_batch.py 50       # 50 examples
    python3 test_batch.py 10 test  # use data/test_real.tsv
"""

import sys

from src.infer import load_model, translate

from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"


def read_pairs(split: str = "val"):
    for name in (f"{split}_real.tsv", f"{split}.csv"):
        path = DATA_DIR / name
        if not path.exists():
            continue
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" in line:
                    src, tgt = line.split("\t", 1)
                    if src.strip() and tgt.strip():
                        pairs.append((src.strip(), tgt.strip()))
        return pairs
    raise FileNotFoundError(f"no {split} data found in {DATA_DIR}")


def main(num_pairs: int = 20, split: str = "val"):
    model, tok = load_model()
    pairs = read_pairs(split)

    print(f"Loaded {len(pairs)} {split} pairs; showing {min(num_pairs, len(pairs))}\n")
    print("=" * 90)

    for i, (src, tgt) in enumerate(pairs[:num_pairs], 1):
        pred = translate(model, tok, src)
        print(f"{i:3d}. ASH       : {src}")
        print(f"     GRB true  : {tgt}")
        print(f"     GRB model : {pred}")
        print()

    print("=" * 90)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sp = sys.argv[2] if len(sys.argv) > 2 else "val"
    main(n, sp)
