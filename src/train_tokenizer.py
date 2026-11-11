# -*- coding: utf-8 -*-
"""
train_tokenizer.py
Обучает BPE-токенайзер на объединённом корпусе ашхарабара и грабара.
Сохраняет модель в tokenizer/sp_model.json
"""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

BASE = Path(r"D:\armenian_nlp")
DATA_DIR = BASE / "data"
TOK_DIR = BASE / "tokenizer"
TOK_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
TOK_OUT = TOK_DIR / "sp_model.json"

def read_texts(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            src, tgt = line.split("\t", 1)
            yield src
            yield tgt

def main():
    if not TRAIN.exists():
        print(f"[!] Файл {TRAIN} не найден. Сначала запустите prepare_data.py")
        return

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=12000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"],
        min_frequency=2,
        show_progress=True,
    )

    texts = list(read_texts(TRAIN))
    print(f"Обучаю токенайзер на {len(texts)} строках...")
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(str(TOK_OUT))
    print(f"[OK] Токенайзер сохранён: {TOK_OUT}")

    test = "Բարև,=eû Armenia"
    enc = tokenizer.encode(test)
    print(f"Тест: {test}")
    print(f"Токены: {enc.tokens}")
    print(f"IDs: {enc.ids}")

if __name__ == "__main__":
    main()
