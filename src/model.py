# -*- coding: utf-8 -*-
"""
model.py
Encoder-Decoder Transformer с нуля для перевода ашхарабар → grabar.
Исправленная версия — ключи совпадают с чекпоинтом, сохранённым на Kaggle.
"""

import math
import torch
import torch.nn as nn


PAD_IDX = 3
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 0


class PosEnc(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransMT(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 256,
        nhead: int = 8,
        layers: int = 4,
        ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pe = PosEnc(d, dropout)
        self.trans = nn.Transformer(d, nhead, layers, layers, ff, dropout, batch_first=True)
        self.gen = nn.Linear(d, vocab)
        self.d = d
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        s = self.pe(self.emb(src) * math.sqrt(self.d))
        t = self.pe(self.emb(tgt) * math.sqrt(self.d))
        out = self.trans(
            s,
            t,
            src_key_padding_mask=(src == PAD_IDX),
            tgt_mask=torch.tril(torch.ones(tgt.size(1), tgt.size(1))).bool().to(src.device),
            tgt_key_padding_mask=(tgt == PAD_IDX),
            memory_key_padding_mask=(src == PAD_IDX),
        )
        return self.gen(out)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int = 100, device: str = "cpu"):
        self.eval()
        src = src.to(device)
        memory = self.trans.encoder(
            self.pe(self.emb(src) * math.sqrt(self.d)),
            src_key_padding_mask=(src == 2),
        )
        ys = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            T = ys.size(1)
            causal_mask = torch.tril(torch.ones(T, T, device=src.device)).bool()
            ys_pad_mask = (ys == PAD_IDX)

            out = self.trans.decoder(
                self.pe(self.emb(ys) * math.sqrt(self.d)),
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=ys_pad_mask,
                memory_key_padding_mask=(src == PAD_IDX),
            )
            logits = self.gen(out[:, -1])
            next_token = logits.argmax(dim=-1).item()
            ys = torch.cat(
                [ys, torch.full((1, 1), next_token, dtype=torch.long, device=device)],
                dim=1,
            )
            if next_token == EOS_IDX:
                break

        return ys[0]
