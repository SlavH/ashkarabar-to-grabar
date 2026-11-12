#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py - evaluate the model on the held-out test split (chrF + BLEU).

Usage:
    pip install sacrebleu
    python3 eval_metrics.py            # data/test_real.tsv
    python3 eval_metrics.py val        # data/val_real.tsv
"""

import sys
from pathlib import Path

from src.infer import load_model, translate

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"


def read_pairs(split: str):
    path = DATA_DIR / f"{split}_real.tsv"
    if not path.exists():
        path = DATA_DIR / f"{split}.csv"
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                src, tgt = line.split("\t", 1)
                if src.strip() and tgt.strip():
                    pairs.append((src.strip(), tgt.strip()))
    return pairs


def main(split: str = "test"):
    try:
        import sacrebleu
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        return

    model, tok = load_model()
    pairs = read_pairs(split)
    sources = [p[0] for p in pairs]
    references = [p[1] for p in pairs]
    predictions = [translate(model, tok, s) for s in sources]

    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print(f"Split: {split} ({len(sources)} pairs)")
    print(f"chrF: {chrf.score:.2f}")
    print(f"BLEU: {bleu.score:.2f}")
    print()
    for i in range(min(5, len(sources))):
        print(f"  ASH : {sources[i]}")
        print(f"  GOLD: {references[i]}")
        print(f"  PRED: {predictions[i]}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
