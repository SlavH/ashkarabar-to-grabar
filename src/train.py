# -*- coding: utf-8 -*-
"""
train.py
Цикл обучения Transformer MT с чекпоинтами на disk D:
"""

import math
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from model import TransMT

BASE = Path(r"D:\armenian_nlp")
DATA_DIR = BASE / "data"
CKPT_DIR = BASE / "checkpoints"
TOK_DIR = BASE / "tokenizer"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train.csv"
VAL = DATA_DIR / "val.csv"
TOK_MODEL = TOK_DIR / "sp_model.json"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
CLIP = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# Special token indices (must match tokenizer and model)
PAD_IDX = 3
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 0


class ParallelDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Tokenizer, max_len: int = MAX_LEN):
        self.pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                src, tgt = line.split("\t", 1)
                self.pairs.append((src, tgt))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = [BOS_IDX] + self.tokenizer.encode(src).ids + [EOS_IDX]
        tgt_ids = [BOS_IDX] + self.tokenizer.encode(tgt).ids + [EOS_IDX]
        src_ids = src_ids[: self.max_len]
        tgt_ids = tgt_ids[: self.max_len]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    if not TRAIN.exists() or not VAL.exists():
        print("[!] Данные не найдены. Запустите prepare_data.py")
        return
    if not TOK_MODEL.exists():
        print("[!] Токенайзер не найден. Запустите train_tokenizer.py")
        return

    tokenizer = Tokenizer.from_file(str(TOK_MODEL))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Размер словаря: {vocab_size}")

    train_ds = ParallelDataset(TRAIN, tokenizer)
    val_ds = ParallelDataset(VAL, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = TransMT(vocab_size=vocab_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)

    best_val = float("inf")
    start_epoch = 1

    latest = CKPT_DIR / "last.pt"
    if latest.exists():
        print(f"Найден чекпоинт {latest}, продолжаю обучение...")
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Продолжаю с эпохи {start_epoch}, лучший val={best_val:.4f}")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        dt = time.time() - t0

        print(
            f"Эпоха {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ppl={math.exp(val_loss):.2f} | время={dt:.1f}с"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            CKPT_DIR / "last.pt",
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                CKPT_DIR / "best.pt",
            )
            print(f"  >>> Новый лучший чекпоинт (val_loss={best_val:.4f})")

    print(f"\n[OK] Обучение завершено. Лучший val_loss={best_val:.4f}")
    print(f"Чекпоинты в {CKPT_DIR}")


if __name__ == "__main__":
    main()
