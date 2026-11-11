#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch.py - Batch test on validation set

Usage:
    python test_batch.py
    python test_batch.py 50   # test first 50 pairs
"""

from pathlib import Path
import torch
from tokenizers import Tokenizer

from src.model import TransMT

_candidate = Path(r"D:\armenian_nlp")
BASE = _candidate if _candidate.exists() else Path(__file__).resolve().parent
CKPT_DIR = BASE / "checkpoints"
TOK_DIR = BASE / "tokenizer"
DATA_DIR = BASE / "data"

PAD_IDX = 3
BOS_IDX = 1
EOS_IDX = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    tok = Tokenizer.from_file(str(TOK_DIR / "sp_model.json"))
    vocab_size = tok.get_vocab_size()
    model = TransMT(vocab=vocab_size).to(DEVICE)
    ckpt = torch.load(CKPT_DIR / "armenian_mt.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok


def translate(model: TransMT, tok: Tokenizer, text: str, max_len: int = 100) -> str:
    ids = [BOS_IDX] + tok.encode(text).ids + [EOS_IDX]
    ids = ids[:max_len]
    src = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out_ids = model.greedy_decode(src, max_len=max_len, device=DEVICE)

    ids_out = out_ids.tolist()
    if ids_out[0] == BOS_IDX:
        ids_out = ids_out[1:]
    if EOS_IDX in ids_out:
        ids_out = ids_out[: ids_out.index(EOS_IDX)]

    return tok.decode(ids_out)


def main(num_pairs: int = 20):
    model, tok = load_model()

    pairs = []
    with open(DATA_DIR / "val.csv", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                src, tgt = line.split("\t", 1)
                pairs.append((src, tgt))

    print(f"Loaded {len(pairs)} validation pairs")
    print(f"Showing first {min(num_pairs, len(pairs))}\n")
    print("=" * 90)

    shown = 0
    for i, (src, tgt) in enumerate(pairs, 1):
        if shown >= num_pairs:
            break
        pred = translate(model, tok, src)
        print(f"{i:3d}. ASH : {src}")
        print(f"    GRB (true): {tgt}")
        print(f"    GRB (pred): {pred}")
        print()
        shown += 1

    print("=" * 90)
    print(f"Displayed {shown} of {len(pairs)} validation pairs")


if __name__ == "__main__":
    import sys

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(num)
