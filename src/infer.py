# -*- coding: utf-8 -*-
"""
infer.py
Modern Armenian (ashkharabar) -> Classical Armenian (grabar) inference.

The checkpoint carries both the weights and the architecture config, so any
model trained by kaggle_train.ipynb loads here without extra flags.
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.model import BOS_IDX, EOS_IDX, TransMT

BASE = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE / "checkpoints"
TOK_DIR = BASE / "tokenizer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def load_tokenizer(path: Path | None = None) -> Tokenizer:
    path = Path(path) if path else TOK_DIR / "sp_model.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {path}\n"
            "Copy tokenizer/sp_model.json from the Kaggle output."
        )
    return Tokenizer.from_file(str(path))


def load_model(checkpoint_path: Path | None = None, tokenizer: Tokenizer | None = None,
               device: str = DEVICE) -> tuple[TransMT, Tokenizer]:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else CKPT_DIR / "armenian_mt.pt"
    tokenizer = tokenizer or load_tokenizer()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = ckpt.get("vocab", tokenizer.get_vocab_size())
    model = TransMT.from_config(vocab, ckpt.get("config") or {}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"Loaded {checkpoint_path.name}: vocab={vocab}, config={ckpt.get('config')}, "
        f"val_loss={ckpt.get('val_loss', '?')}, epoch={ckpt.get('epoch', '?')}"
    )
    return model, tokenizer


def translate(model: TransMT, tokenizer: Tokenizer, text: str,
              max_len: int = MAX_LEN, device: str = DEVICE) -> str:
    ids = [BOS_IDX] + tokenizer.encode(text).ids + [EOS_IDX]
    ids = ids[:max_len]
    src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out_ids = model.greedy_decode(src, max_len=max_len, device=device).tolist()

    if out_ids and out_ids[0] == BOS_IDX:
        out_ids = out_ids[1:]
    if EOS_IDX in out_ids:
        out_ids = out_ids[: out_ids.index(EOS_IDX)]

    return tokenizer.decode(out_ids).strip()


def main():
    parser = argparse.ArgumentParser(description="Translate Armenian: ashkharabar -> grabar")
    parser.add_argument("--text", required=True, help="text in modern Armenian")
    parser.add_argument("--checkpoint", default=str(CKPT_DIR / "armenian_mt.pt"))
    args = parser.parse_args()

    model, tokenizer = load_model(Path(args.checkpoint))
    result = translate(model, tokenizer, args.text)
    print(f"\nInput  (ASH): {args.text}")
    print(f"Output (GRB): {result}")


if __name__ == "__main__":
    main()
