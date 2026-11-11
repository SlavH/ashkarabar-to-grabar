#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_one.py - Translate a single sentence from Ашхарабар to Грабар

Usage:
    python test_one.py --text "Բարև, աշխարհ"
    python test_one.py --text "Ես սիրում եմ հայերենը" --checkpoint checkpoints/armenian_mt.pt
"""

import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from src.model import TransMT

_candidate = Path(r"D:\armenian_nlp")
BASE = _candidate if _candidate.exists() else Path(__file__).resolve().parent
CKPT_DIR = BASE / "checkpoints"
TOK_DIR = BASE / "tokenizer"

PAD_IDX = 3
BOS_IDX = 1
EOS_IDX = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path, tokenizer: Tokenizer) -> TransMT:
    vocab_size = tokenizer.get_vocab_size()
    model = TransMT(vocab=vocab_size).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Model loaded from {checkpoint_path.name} "
        f"(epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})"
    )
    return model


def translate(model: TransMT, tokenizer: Tokenizer, text: str, max_len: int = 100) -> str:
    ids = [BOS_IDX] + tokenizer.encode(text).ids + [EOS_IDX]
    ids = ids[:max_len]
    src = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        out_ids = model.greedy_decode(src, max_len=max_len, device=DEVICE)

    ids_out = out_ids.tolist()
    if ids_out[0] == BOS_IDX:
        ids_out = ids_out[1:]
    if EOS_IDX in ids_out:
        ids_out = ids_out[: ids_out.index(EOS_IDX)]

    return tokenizer.decode(ids_out)


def main():
    parser = argparse.ArgumentParser(description="Translate Armenian (ashkharabar -> grabar)")
    parser.add_argument("--text", required=True, help="Text in ashkharabar to translate")
    parser.add_argument(
        "--checkpoint",
        default=str(CKPT_DIR / "armenian_mt.pt"),
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    tok_path = TOK_DIR / "sp_model.json"
    if not tok_path.exists():
        print(f"ERROR: Tokenizer not found at {tok_path}")
        print("Run train_tokenizer.py first, or copy tokenizer from Kaggle output.")
        return

    tokenizer = Tokenizer.from_file(str(tok_path))
    model = load_model(Path(args.checkpoint), tokenizer)

    result = translate(model, tokenizer, args.text)
    print(f"\nInput  (ASH): {args.text}")
    print(f"Output (GRB): {result}")


if __name__ == "__main__":
    main()
