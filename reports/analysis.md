# Error & Coverage Analysis (offline)

Snapshot date: 2026-11-14 (UTC). Based on frozen `data/test_real.tsv` (n=108), current splits, and saved sample outputs in `reports/nllb_samples.txt` (v9-era inference samples) and metric logs. No local model inference was run.

## 1. Evaluation setup (current best)
- **Model:** `facebook/nllb-200-distilled-600M` + LoRA adapter `checkpoints/nllb_lora_grabar/` (v9: r=64, α=128, dropout 0.05, targets q/k/v/out/fc1/fc2), trained on `train_v3` (2852) / `val_v3` (256), max_len 160.
- **Frozen test:** `data/test_real.tsv`, 108 verse-level pairs (ashkharabar → grabar), biblical domain.
- **Current best (E17, adopted):** **chrF 54.80 / BLEU 24.06** with decoding `num_beams=4, length_penalty=1.05, no_repeat_ngram_size=4, repetition_penalty=1.4, min_new_tokens=15`.
- **Progression (same test):** v9 baseline decode 53.84 / 22.86 → 54.80 / 24.06 (+0.96 chrF, +1.20 BLEU) via decoding search (E5/E9/E15/E17). Training/seed/data variants evaluated so far have not improved chrF over the v9 adapter.

## 2. Data profile
| Split | Pairs | Src mean words | Tgt mean words |
|---|---:|---:|---:|
| train_v5 | 2872 | ~18.1 | ~18.5 |
| val_v5 | 256 | (same regime as v4) | |
| test_real (frozen) | 108 | ~18.3 | ~19.1 |

- Domain: predominantly biblical (Zohrab/Genesis–Exodus passages, NT lections) plus Abeghyan dasagirq/lectionary material. Grabar targets consistently use classical orthography/features (եւ/ւ, classical punctuation/particles) across the test set.
- Recent addition: `train_v5` = `train_v4` + 6 verified pairs (`data/dasagirq_extra_nt_matt579.tsv`, Matt 7:20,29; 9:11,20–22), deduped and leak-checked. v5 is prepared for future training (E19-v24 script ready).

## 3. Observed strengths (from sampled outputs)
- **Core meaning and clause structure** are often preserved for frequent biblical constructions (dialogue/reporting frames like “Եւ ասէ…”, commandments, narrative sequences).
- **Common formulaic vocabulary** maps reasonably well (e.g. recurring divine names/epithets, frequent verbs of speaking/seeing/going) when present in training distribution.
- Some examples achieve high surface overlap with gold (close register and phrasing), indicating the adapter captures salient grabar patterns when input is in-distribution.

## 4. Recurring error patterns (qualitative, sample-based)
These are heuristic observations from `reports/nllb_samples.txt` and spot checks; a full per-example scored re-eval with current decoding should confirm magnitudes.

1. **Lexical/register choice (classical vs modern/colloquial):** occasional modern or simplified lexemes/morphology where gold uses a more classical form; also paraphrases that are semantically close but not idiomatic grabar (e.g. different verb/noun choice than the reference).
2. **Morphology and case/inflection fidelity:** endings, agreement, and nominal/verbal forms sometimes drift from the reference (small edits that affect BLEU/chrF and stylistic correctness).
3. **Proper names and transliteration variants:** biblical names may appear in a plausible but non-reference form/spelling; consistency with the specific grabar source tradition varies.
4. **Content completeness:** a minority of outputs are shorter than gold (omissions/compression) or occasionally longer/repetitive; tuned decoding (no_repeat_ngram, repetition_penalty, min_new_tokens) mitigates repetition but does not fully solve omissions or additions.
5. **Rare or out-of-distribution constructions:** verses with less common syntax, rare terms, or patterns underrepresented in ~3k training pairs are more error-prone.
6. **Reference dependence:** there can be multiple acceptable grabar renderings; string metrics may penalize valid alternatives (limits absolute interpretability of chrF/BLEU at this scale).

## 5. Constraints limiting “perfect” translation
- **Data scarcity & domain skew:** ~2.9k train pairs, heavily biblical/lectionary; limited coverage of broader registers and constructions.
- **Model/tokenization:** NLLB-200-distilled-600M has no dedicated Classical Armenian tokenization target (both sides mapped to `hye_Armn`), which caps fine-grained classical control.
- **Compute/eval budget:** Kaggle-only; weekly GPU quota can block iteration (e.g. E18 not run). Test set is small (108), so small metric deltas should be treated cautiously.

## 6. Recommendations (next iterations)
- **Data (highest leverage):** source and *manually verify* additional parallel ashkharabar↔grabar pairs with strict verse/line alignment and leakage checks (extend beyond current dasagirq/Zohrab coverage). Even small, clean additions (v5-style) are safer than noisy bulk data.
- **Decoding (low-risk):** when GPU quota allows, run **E18-decode5** (nr/min_new_tokens around current best) and only adopt if chrF > 54.80 on frozen test.
- **Training (targeted):** try **E19-v24** on `train_v5` (prepared) and consider carefully scoped variants (e.g. curriculum/weighting by source, mild augmentation only if quality-controlled); adopt only on frozen-test chrF gain.
- **Analysis:** generate and archive predictions with the *current* decode defaults for all 108 test items to enable precise per-example error tagging (names, morphology, omission/addition) and to track regressions.

## 7. Status
- Best deployed: v9 adapter + E17 decoding (chrF 54.80 / BLEU 24.06). E18 prepared but not run (quota). E19-v24 prepared on train_v5. Repo experiments log: `reports/experiments.md`.
