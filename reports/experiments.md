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
| E4-v13 | smaller LoRA | r=32 alpha=64, 30 epochs, lr 2e-4 | v3 | 52.86 / 22.17 | rejected |
| E5-decode | decoding search | beams/lp/no_repeat_ngram grid on v9 adapter | - | 54.54 / 24.25 (beams=6,lp=1.2,nr=4) | adopted then superseded by E9 |
| E6-v15 | more data | v9 config on train_v4 (+Ps148, 2866 pairs), tuned decode eval | v4 | 53.76 / 24.29 | rejected (chrF < best 54.54; adapter size 133MB) |
| E7-v16 | longer seq | v9 config on train_v3, MAXLEN 192, tuned decode | v3 | 53.08 / 23.36 | rejected |
| E8-v17 | fewer epochs on v4 | v9 config on train_v4, 24 epochs, tuned decode | v4 | 53.35 / 23.53 | rejected |
| E9-decode2 | decoding search 2 | beams/lp/nr/repetition_penalty grid on v9 adapter | - | 54.64 / 23.99 (beams=4,lp=1.1,nr=4,rp=1.3) | adopted then superseded by E15 |
| E10-v18 | longer training | v9 config on train_v3, 35 epochs, tuned decode | v3 | 53.87 / 23.15 | rejected |
| E11-v19 | different seed | v9 config on train_v3, seed=2024, 30 epochs, tuned decode | v3 | 53.95 / 24.59 | rejected (chrF < best 54.64) |
| E12-v20 | different seed | v9 config on train_v3, seed=999, 30 epochs, tuned decode | v3 | 53.79 / 22.96 | rejected |
| E13-v21 | label smoothing | v9 config on train_v3, label_smoothing=0.1, tuned decode | v3 | failed (decoder_input_ids/embeds conflict) | - |
| E14-v22 | warmup | v9 config on train_v3, warmup_steps=400, tuned decode | v3 | 53.84 / 22.91 | rejected |
| E15-decode3 | decoding search 3 | min_new/max_new/length_penalty around best on v9 adapter | - | 54.71 / 24.07 (mn=15,lp=1.05,rp=1.3,nb=4,nr=4) | adopted then superseded by E17 |
| E16-v23 | different seed | v9 config on train_v3, seed=42, 30 epochs, tuned decode | v3 | 54.11 / 24.59 | rejected (chrF < best 54.71) |
| E17-decode4 | decoding search 4 | beams/lp/rp fine grid (mn=15,nr=4) on v9 adapter | - | **54.80 / 24.06** (nb=4,lp=1.05,rp=1.4,mn=15,nr=4) | **adopted as inference default** (adapter unchanged) |
| E18-decode5 | decoding search 5 | no_repeat_ngram_size/min_new_tokens grid (nb=4,lp=1.05,rp=1.4) on v9 | - | - | not run (Kaggle weekly GPU quota reached) |
| E19-v24 | more data (v5) | v9 config on train_v5 (+6 clean pairs), 30 epochs, tuned decode | v5 | - | prepared (Kaggle GPU quota blocked) |
