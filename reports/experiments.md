# Experiments log

Baseline (adopted): **v9** — chrF 53.84 / BLEU 22.86, n=108 test (frozen). Config: NLLB-200-distilled-600M, LoRA r=64/α=128 targets q,k,v,out,fc1,fc2, dropout 0.05, 30 epochs, lr 2e-4 cosine warmup 200, Adafactor, fp16, batch 8×accum 2, maxlen 160. Data: train_v3 (2852) / val_v3 (256).

| Run | Hypothesis | Changes | Data | Result (test chrF/BLEU) | Decision |
|---|---|---|---|---|---|
| v9 | longer training | 30 epochs, clean data | v3 | 53.84 / 22.86 | **adopted** |
| v10 | augmentation helps | +2100 char-swap noisy pairs | v3_aug | 52.53 / 22.32 | rejected (noise diluted clean signal) |

## Data search (Hour 0.5–2.5)
- `Աստուածաշունչ (1805)` on hy.wikisource: only Genesis and Exodus transcribed. No additional books.
- `ԱՍՏՈՒԱԾԱՇՈՒՆՉ/ՂԵՒՏԱՑՈՑ` (Leviticus), `/ՀՌՈՒԹ` (Ruth): modern Armenian (19th c.), not Grabar — unusable as targets.
- `k-mktr/armenian_bible_hy`: only books 1,2,40,41,42,43 (Gen,Ex,Matt,Mark,Luke,John).
- `Գրաբարի դասագիրք` subpages: 45 pages total; correspond to already-fetched numbered pages / alternate titles, no new bilingual pairs found.
- HF datasets checked: `catherinearnett/classical_armenian` (Grabar Zohrab, non-parallel/PD mix), `catherinearnett/classical_armenian_pd` (19th-c modern lit, mislabeled), `nomikos-project/armenian-manuscript-htr` (HTR images) — no usable parallel modern↔grabar text.
- **Conclusion:** no new reliable parallel data identified in this pass. Proceeding with modeling experiments on v3 data.
| E2-v11 | regularization | 24 epochs, lora_dropout 0.1 | v3 | 53.53 / 23.35 | rejected (chrF < v9) |
| E3-v12 | lr/warmup | lr 1e-4, warmup 300, 30 epochs, dropout 0.05 | v3 | 51.09 / 20.53 | rejected (too slow LR) |
| E4-v13 | smaller LoRA | r=32 alpha=64, 30 epochs, lr 2e-4 | v3 | running | - |
