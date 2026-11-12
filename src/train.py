# -*- coding: utf-8 -*-
"""
train.py - local training loop (CPU/GPU) for Modern Armenian -> Classical Armenian.

The same recipe runs on Kaggle (see kaggle_train.ipynb). Use small
--epochs values for smoke tests on CPU; real training belongs on a GPU.

    python3 -m src.train --epochs 200
    python3 -m src.train --epochs 2 --limit 50     # quick sanity check
"""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset

from src.model import BOS_IDX, EOS_IDX, PAD_IDX, TransMT

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
CKPT_DIR = BASE / "checkpoints"
TOK_DIR = BASE / "tokenizer"

VOCAB_SIZE = 2400
MAXLEN = 160
BATCH = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_pairs(path: Path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            a, b = line.split("\t", 1)
            a, b = a.strip(), b.strip()
            if len(a) > 10 and len(b) > 10:
                pairs.append((a, b))
    return pairs


def build_tokenizer(pairs) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator([x for p in pairs for x in p], trainer)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK_DIR / "sp_model.json"))
    return tok


class PairDS(Dataset):
    def __init__(self, pairs, tok):
        self.pairs, self.tok = pairs, tok

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        ai = [BOS_IDX] + self.tok.encode(a).ids[: MAXLEN - 2] + [EOS_IDX]
        bi = [BOS_IDX] + self.tok.encode(b).ids[: MAXLEN - 2] + [EOS_IDX]
        return torch.tensor(ai), torch.tensor(bi)


def collate(batch):
    src, tgt = zip(*batch)
    return (
        nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX),
        nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="cap on training pairs (smoke tests)")
    args = ap.parse_args()

    train_pairs = read_pairs(DATA_DIR / "train_real.tsv")
    val_pairs = read_pairs(DATA_DIR / "val_real.tsv")
    if args.limit:
        train_pairs = train_pairs[: args.limit]
        val_pairs = val_pairs[: max(8, args.limit // 5)]

    random.seed(1234)
    torch.manual_seed(1234)

    tok_path = TOK_DIR / "sp_model.json"
    tok = Tokenizer.from_file(str(tok_path)) if tok_path.exists() else build_tokenizer(train_pairs)
    vocab = tok.get_vocab_size()

    print(f"device={DEVICE} vocab={vocab} train={len(train_pairs)} val={len(val_pairs)}")

    dl_train = DataLoader(PairDS(train_pairs, tok), batch_size=BATCH, shuffle=True,
                          collate_fn=collate)
    dl_val = DataLoader(PairDS(val_pairs, tok), batch_size=64, shuffle=False,
                        collate_fn=collate)

    model = TransMT(vocab=vocab).to(DEVICE)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=max(1, args.epochs * len(dl_train)),
        pct_start=0.08, anneal_strategy="cos")

    def batch_loss(s, t):
        logits = model(s, t[:, :-1])
        return crit(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for s, t in dl_val:
            s, t = s.to(DEVICE), t.to(DEVICE)
            loss = batch_loss(s, t)
            if torch.isfinite(loss):
                tot += loss.item()
                n += 1
        model.train()
        return tot / max(1, n)

    best, best_state, best_ep = float("inf"), None, 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for s, t in dl_train:
            s, t = s.to(DEVICE), t.to(DEVICE)
            loss = 0.5 * (batch_loss(s, t) + batch_loss(t, s))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1

        vl = evaluate()
        msg = (f"epoch {ep:3d}/{args.epochs} train={tot / max(1, nb):.4f} "
               f"val={vl:.4f} lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            msg += " *best*"
            torch.save({"model": best_state, "vocab": vocab, "config": model.config,
                        "val_loss": best, "epoch": ep}, CKPT_DIR / "armenian_mt.pt")
        print(msg, flush=True)

    print(f"\nbest val={best:.4f} (epoch {best_ep}); ppl={math.exp(best):.2f}")
    print(f"checkpoint: {CKPT_DIR / 'armenian_mt.pt'}")


if __name__ == "__main__":
    main()
