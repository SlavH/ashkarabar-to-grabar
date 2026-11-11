# -*- coding: utf-8 -*-
"""
infer.py
Перевод с ашхарабара на grabar через обученную модель.
"""

import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from src.model import TransMT

BASE = Path(r"D:\armenian_nlp") if Path(r"D:\armenian_nlp").exists() else Path(__file__).resolve().parent
CKPT_DIR = BASE / "checkpoints"
TOK_DIR = BASE / "tokenizer"

PAD_IDX = 3
BOS_IDX = 1
EOS_IDX = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


def load_model(checkpoint_path: Path, tokenizer):
    vocab_size = tokenizer.get_vocab_size()
    model = TransMT(vocab=vocab_size).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"Модель загружена из {checkpoint_path} "
        f"(эпоха={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')})"
    )
    return model


def translate(model: TransMT, tokenizer: Tokenizer, text: str, max_len: int = MAX_LEN) -> str:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Текст на ашхарабаре для перевода")
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "armenian_mt.pt"), help="Путь к чекпоинту")
    args = parser.parse_args()

    tok_path = TOK_DIR / "sp_model.json"
    if not tok_path.exists():
        print(f"[!] Токенайзер не найден: {tok_path}")
        return

    tokenizer = Tokenizer.from_file(str(tok_path))
    model = load_model(Path(args.checkpoint), tokenizer)

    result = translate(model, tokenizer, args.text)
    print(f"\nВход (ашхарабар): {args.text}")
    print(f"Выход (грабар):   {result}")


if __name__ == "__main__":
    main()
