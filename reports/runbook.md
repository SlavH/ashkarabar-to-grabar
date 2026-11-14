# Post-Quota Runbook (Kaggle)

**Current best (adopted):** v9 adapter `checkpoints/nllb_lora_grabar/` + decode  
`num_beams=4, length_penalty=1.05, no_repeat_ngram_size=4, repetition_penalty=1.4, min_new_tokens=15`  
→ **chrF 54.80 / BLEU 24.06** on frozen `data/test_real.tsv` (n=108).  
**Adoption rule:** change defaults/adapter only if **chrF > 54.80** on the frozen test. Always keep `test_real.tsv` untouched and avoid train/val/test leakage.

## Kernel
- Slug: `slavhayrapetyan/armenian-mt-nllb` (T4, internet on), metadata: `kaggle_nllb/kernel-metadata.json`
- Push: `cd kaggle_nllb && kaggle kernels push -p .`
- Status: `kaggle kernels status slavhayrapetyan/armenian-mt-nllb`
- Output: `kaggle kernels output slavhayrapetyan/armenian-mt-nllb -p <outdir>`
- Note: `kaggle_nllb/nllb_train.ipynb` currently holds the **E18-decode5** draft. Regenerate it from a builder before running a different experiment.

## Prepared experiments (run one at a time)
| ID | Builder | What it does | Data | Notes |
|---|---|---|---|---|
| E18-decode5 | `build_nllb_eval_decode5.py` | Decode search: nr × min_new_tokens (nb=4, lp=1.05, rp=1.4) on v9 adapter | - | Eval-only, faster than training. Writes `metrics_decode5.txt`. |
| E19-v24 | `build_nllb_v24.py` | Train v9 config, 30 ep, seed 42, tuned decode eval | train_v5 (2872) | Writes `metrics_nllb.txt`, adapter, samples. |
| E20-v25 | `build_nllb_v25.py` | Train v9 config, 35 ep, seed 42, tuned decode eval | train_v5 | Longer training. |
| E21-v26 | `build_nllb_v26.py` | Train v9 config, lr=1.5e-4, 30 ep, seed 42, tuned decode eval | train_v5 | Lower LR. |
| E22-v27 | `build_nllb_v27.py` | Train v9 config, 30 ep, seed=999, tuned decode eval | train_v5 | Different seed. |

Generate notebook: `python3 build_nllb_v24.py` (overwrites `kaggle_nllb/nllb_train.ipynb`), then push.

## After each run
1. Download outputs; read metrics (and log if present). Verify completeness.
2. **If chrF > 54.80** (decode-only): update decode defaults in `translate_nllb.py`, `demo.py`, `colab_demo.ipynb`, `README.md`; log in `reports/experiments.md` as adopted; commit & push.
3. **If chrF > 54.80** (training): replace `checkpoints/nllb_lora_grabar/` (and tokenizer if changed) with the new adapter, update defaults/docs/samples as needed, keep a backup of previous best, log adopted; commit & push.
4. **If not improved / failed:** log result/reason in `reports/experiments.md`, keep previous best deployed; commit & push. Clean local kaggle output dirs.
5. Do not push large unnecessary artifacts; adapter is ~67 MB (already above GitHub’s recommended 50 MB).

## Data
- `train_v5.tsv` (2872), `val_v5.tsv` (256), `test_real.tsv` (108, frozen). Source additions tracked in `reports/experiments.md` (v4: Psalm 148; v5: 6 verified Matt 5/7/9 pairs).
- Any new data must be manually/strictly aligned, deduped against all splits, and checked for leakage before creating `train_v6`.
