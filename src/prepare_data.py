# -*- coding: utf-8 -*-
"""
prepare_data.py
Очистка и разбивка параллельного корпуса ашхарабар → grabar.
Ожидается входной файл data/raw_pairs.txt с форматом:
ашхарабар\tграбар
"""

from pathlib import Path
import re
import random

BASE = Path(r"D:\armenian_nlp")
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW = DATA_DIR / "raw_pairs.txt"
TRAIN = DATA_DIR / "train.csv"
VAL = DATA_DIR / "val.csv"

ARMENIAN_RE = re.compile(r"[\u0530-\u058F\s\d\.,!?;:'\"\(\)-–—…]+")

def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ARMENIAN_RE.match(ch) or ch in " .,!?;:'\"()–—…")
    return text.strip()

def load_pairs(path: Path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            src, tgt = line.split("\t", 1)
            src = clean(src)
            tgt = clean(tgt)
            if len(src) == 0 or len(tgt) == 0:
                continue
            if len(src) > 200 or len(tgt) > 200:
                continue
            pairs.append((src, tgt))
    return pairs

def main():
    if not RAW.exists():
        print(f"[!] Файл {RAW} не найден.")
        print("Создайте его с форматом: ашхарабар\tграбар (по одной паре на строку)")
        return

    pairs = load_pairs(RAW)
    print(f"Загружено пар после очистки: {len(pairs)}")

    if len(pairs) < 100:
        print("[!] Слишком мало данных. Нужно минимум 500–1000 пар для обучения.")
        return

    random.seed(42)
    random.shuffle(pairs)

    n_val = max(500, int(len(pairs) * 0.1))
    train_pairs = pairs[n_val:]
    val_pairs = pairs[:n_val]

    with open(TRAIN, "w", encoding="utf-8") as f:
        for s, t in train_pairs:
            f.write(f"{s}\t{t}\n")

    with open(VAL, "w", encoding="utf-8") as f:
        for s, t in val_pairs:
            f.write(f"{s}\t{t}\n")

    print(f"[OK] train.csv: {len(train_pairs)} пар")
    print(f"[OK] val.csv: {len(val_pairs)} пар")
    print(f"Файлы сохранены в {DATA_DIR}")

if __name__ == "__main__":
    main()
