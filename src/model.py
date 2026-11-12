# -*- coding: utf-8 -*-
"""
model.py
Transformer encoder-decoder (from scratch) for Modern Armenian -> Classical Armenian.

The architecture is fully described by a config dict that is stored inside the
checkpoint, so training (Kaggle) and inference (local) always agree.
"""

import math

import torch
import torch.nn as nn

# token ids produced by the tokenizer (special tokens are declared in this order)
UNK_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
PAD_IDX = 3

DEFAULT_CONFIG = dict(
    d=384,
    nhead=6,
    layers=4,
    ff=1024,
    dropout=0.2,
    max_len=160,
    norm_first=True,
)


class PosEnc(nn.Module):
    def __init__(self, d: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransMT(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 384,
        nhead: int = 6,
        layers: int = 4,
        ff: int = 1024,
        dropout: float = 0.2,
        max_len: int = 160,
        norm_first: bool = True,
    ):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.cfg = dict(d=d, nhead=nhead, layers=layers, ff=ff,
                        dropout=dropout, max_len=max_len, norm_first=norm_first)

        self.emb = nn.Embedding(vocab, d, padding_idx=PAD_IDX)
        self.pe = PosEnc(d, dropout, max_len=max(max_len, 512))
        self.trans = nn.Transformer(
            d_model=d,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.gen = nn.Linear(d, vocab)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def config(self) -> dict:
        return dict(self.cfg)

    @classmethod
    def from_config(cls, vocab: int, cfg: dict) -> "TransMT":
        return cls(vocab=vocab, **(cfg or {}))

    def encode(self, src):
        return self.trans.encoder(
            self.pe(self.emb(src) * math.sqrt(self.d)),
            src_key_padding_mask=(src == PAD_IDX),
        )

    def forward(self, src, tgt, memory=None):
        if memory is None:
            memory = self.encode(src)
        T = tgt.size(1)
        causal = torch.triu(torch.ones(T, T, device=tgt.device, dtype=torch.bool), diagonal=1)
        out = self.trans.decoder(
            self.pe(self.emb(tgt) * math.sqrt(self.d)),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt == PAD_IDX),
            memory_key_padding_mask=(src == PAD_IDX),
        )
        return self.gen(out)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int = 128, device: str = "cpu"):
        """Greedy decoding for a single source sequence [1, S] -> token ids (with BOS)."""
        self.eval()
        src = src.to(device)
        memory = self.encode(src)
        ys = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(src, ys, memory=memory)[:, -1]
            nxt = int(logits.argmax(dim=-1).item())
            ys = torch.cat(
                [ys, torch.full((1, 1), nxt, dtype=torch.long, device=device)], dim=1
            )
            if nxt == EOS_IDX:
                break
        return ys[0]
