#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py - Evaluate model with chrF and BLEU scores

Usage:
    pip install sacrebleu
    python eval_metrics.py
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


def main():
    try:
        import sacrebleu
    except ImportError:
        print("Install sacrebleu: pip install sacrebleu")
        return

    model, tok = load_model()

    sources, references, predictions = [], [], []
    with open(DATA_DIR / "val.csv", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                src, tgt = line.split("\t", 1)
                sources.append(src)
                references.append(tgt)
                predictions.append(translate(model, tok, src))

    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print(f"Evaluated {len(sources)} pairs")
    print(f"chrF: {chrf.score:.2f}")
    print(f"BLEU: {bleu.score:.2f}")
    print()

    print("Sample translations:")
    for i in range(min(5, len(sources))):
        print(f"  ASH : {sources[i]}")
        print(f"  GRB (true): {references[i]}")
        print(f"  GRB (pred): {predictions[i]}")
        print()


if __name__ == "__main__":
    main()
