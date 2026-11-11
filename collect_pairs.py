#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_pairs.py - Interactive tool to collect parallel Armenian pairs
Modern Armenian (ashkharabar) -> Classical Armenian (grabar)

Usage:
    python collect_pairs.py --file data/raw_pairs.txt --pairs 50

This will prompt you for each pair and append to raw_pairs.txt.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect parallel Armenian pairs")
    parser.add_argument("--file", default="data/raw_pairs.txt", help="Output file path")
    parser.add_argument("--pairs", type=int, default=10, help="Number of pairs to collect")
    args = parser.parse_args()

    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.pairs} parallel pairs...")
    print("Format: type modern Armenian, then classical Armenian")
    print("Press Ctrl+C to stop early\n")

    with open(path, "a", encoding="utf-8") as f:
        for i in range(args.pairs):
            print(f"\n--- Pair {i + 1}/{args.pairs} ---")
            ashkharabar = input("ASHKHARABAR: ").strip()
            if not ashkharabar:
                print("Skipping empty entry...")
                continue
            grabar = input("GRABAR: ").strip()
            if not grabar:
                print("Skipping empty entry...")
                continue
            f.write(f"{ashkharabar}\t{grabar}\n")
            f.flush()
            print(f"  Saved: {ashkharabar[:40]}... | {grabar[:40]}...")

    print(f"\nDone! Collected pairs saved to {path}")
    print(f"Total lines in file: {sum(1 for _ in open(path, encoding='utf-8'))}")


if __name__ == "__main__":
    main()
