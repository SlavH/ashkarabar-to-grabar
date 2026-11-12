# Armenian MT: Ashkharabar → Grabar

<sup>[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)</sup>
<sup>[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org)</sup>
<sup>[![Kaggle](https://img.shields.io/badge/trained%20on-kaggle%20T4-blue.svg)](https://kaggle.com)</sup>
<sup>[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)</sup>

Neural machine translation from **Modern Armenian (աշխարհաբար, ashkharabar)** to
**Classical Armenian (գրաբար, grabar)** — the literary language of the 5th–19th
centuries, which has almost no parallel NLP resources.

Two models are included:

| Model | Where | Size | chrF | BLEU |
|---|---|---|---|---|
| Transformer trained from scratch | `checkpoints/armenian_mt.pt` | 62 MB | 30.6 | 9.1 |
| NLLB-200 distilled-600M + LoRA (v2) | `checkpoints/nllb_lora_grabar/` (backup) | 37 MB adapter | 51.2 | 22.2 |
| NLLB-200 distilled-600M + LoRA (v3) | (backup) | 67 MB adapter | 51.8 | 21.6 |
| NLLB-200 distilled-600M + LoRA (v9) | `checkpoints/nllb_lora_grabar/` | 67 MB adapter | 53.8 | 22.9 |

Scores are on the same held-out set of 108 verse pairs (chrF/BLEU via
`sacrebleu`, tokenized text, single reference). For a low-resource, morphologically
rich target like grabar these numbers are a starting point, not a solved problem —
see [Limitations](#limitations).

## Example output

Real held-out verses (not seen during training):

```
ASH  : Ողջ ժողովուրդը, տեսնելով վրանի դռան մօտ կանգնած ամպի սիւնը, ոտքի էր կանգնում, իւրաքանչիւրը երկրպագում էր իր վրանի դռան մօտ:
GOLD : Եւ տեսանէր ամենայն ժողովուրդն զսի՛ւն ամպոյն որ կայր առ դուրս խորանին. եւ կայի՛ն ամենայն ժողովուրդն յոտն, եւ երկի՛ր պագանէին իւրաքանչիւր առ դուրս խորանի իւրոյ:
MODEL: Եւ տեսանէին ամենայն ժողովուրդն զսիւն ամպոյն որ կայր առ դրան խորանին, եւ կայր ոտս իւրեանց. եւ երկիր պագա՛յք իւրաքանչիւր առ դուռն խորանին իւրում:

ASH  : Ժողովուրդը միահամուռ պատասխան տուեց ու ասաց. «Այն ամէնը, ինչ ասել է Աստուած, կը կատարենք ու կը հնազանդուենք»:
GOLD : Պատասխանի ետ ժողովուրդն միահամուռ, եւ ասեն. Զամենայն ինչ զոր ասաց Աստուած՝ արասցո՛ւք եւ լուիցուք:
MODEL: Պատասխանի ետ ժողովուրդն միաձայն՝ եւ ասեն. Զամենայն զոր խօսեցաւ Աստուած՝ արասցուք եւ լուիցուք:

# free-form input, outside the Bible domain
ASH  : Աստուած ստեղծեց երկինքն ու երկիրը։
MODEL: Աստուած արար զերկինս եւ զերկիր։

ASH  : Ես սիրում եմ հայերէնը։
MODEL: Ես սիրեմ հայերէնս։
```

## Training data

A **real** parallel corpus, not synthetic text. 2,715 verse-aligned pairs:

| Source | Content |
|---|---|
| hy.wikisource «Աստուածաշունչ (1805)» | Zohrab's 1805 Grabar Bible, Genesis + Exodus, verse-tagged (`{{տուն}}` markers) |
| HF `k-mktr/armenian_bible_hy` | Modern Eastern Armenian Bible (same translation tradition), verse IDs `Gen_1_1` … |

Verses are matched by book/chapter/verse number, cleaned of wiki markup, filtered
(min 25 chars, deduplicated) and split 2,390 train / 217 val / 108 test.

`build_grabar_corpus.py` rebuilds the corpus end-to-end:

```bash
python3 build_grabar_corpus.py     # -> data/parallel_real.tsv + splits
```

## Project layout

```
.
├── build_grabar_corpus.py        # regrabar corpus from hy.wikisource + modern Bible
├── build_notebook.py             # regenerates the from-scratch Kaggle notebook
├── build_nllb_notebook.py        # regenerates the NLLB Kaggle notebook
├── checkpoints/
│   ├── armenian_mt.pt            # from-scratch Transformer (best val epoch 20)
│   └── nllb_lora_grabar/         # LoRA adapter for NLLB-200
├── tokenizer/sp_model.json       # 2,400-token BPE used by the from-scratch model
├── data/                         # train/val/test_real.tsv (modern \t grabar)
├── src/
│   ├── model.py                  # TransMT: config-driven Transformer encoder-decoder
│   ├── train.py                  # local training loop (GPU/CPU)
│   └── infer.py                  # checkpoint loader + greedy decode
├── test_one.py / test_batch.py   # quick inference
├── eval_metrics.py               # chrF + BLEU on a held-out split
├── translate_nllb.py             # NLLB + LoRA inference
├── kaggle_train.ipynb            # from-scratch training (Kaggle T4)
└── kaggle_nllb/nllb_train.ipynb  # NLLB LoRA fine-tuning (Kaggle T4)
```

## Usage

### From-scratch Transformer (fast, CPU-friendly)

```bash
pip install -r requirements.txt

python3 test_one.py --text "Ես սիրում եմ հայերէնը։"
python3 test_batch.py 10 test       # 10 held-out examples
python3 eval_metrics.py test        # chrF / BLEU on the test split
```

### NLLB-200 + LoRA (better quality, needs ~3 GB download)

```bash
python3 translate_nllb.py --text "Աստուած ստեղծեց երկինքն ու երկիրը։"
```

The adapter is in `checkpoints/nllb_lora_grabar/`; the 600M base model
(`facebook/nllb-200-distilled-600M`) is pulled from Hugging Face on first run.

### Training on Kaggle

Both notebooks run on a free T4:

```bash
kaggle kernels push -p .                 # from-scratch run
kaggle kernels push -p kaggle_nllb       # NLLB LoRA run
```

## How the from-scratch model is built

* 15.3M-parameter Transformer encoder–decoder (`d=384`, 6 heads, 4+4 layers,
  `norm_first`, dropout 0.2), 2,400-token byte-level BPE.
* Bidirectional loss: `0.5 × (ASH→GRB + GRB→ASH)`, label smoothing 0.1,
  OneCycle LR, gradient clipping, best-validation checkpointing.
* 200 epochs on 2,390 pairs (≈51 min on a T4). Best validation loss 4.1692 at
  epoch 20; later epochs overfit (train loss 1.27), which is expected with a
  corpus this small.

## How the NLLB model is built

* Base: `facebook/nllb-200-distilled-600M`; LoRA (`r=32`, `alpha=64`) on all four
  attention projections → 14.8M trainable parameters (38 MB adapter).
* 30 epochs, effective batch 16, lr 3e-4 (cosine + warmup), fp16, Adafactor;
  best validation loss 1.4814 at epoch 25, ≈42 min on a single T4.
* Both source and target use the `hye_Armn` language tag — NLLB-200 has no
  separate Classical Armenian token, so the mapping is learned from data.
* Beam search (4 beams, `no_repeat_ngram_size=3`) at inference.

## Limitations

* **Domain:** training data is still biblical (Genesis, Exodus, plus the gospel
  lections and psalms in Abeghyan's 1916 «Գրաբարի դասագիրք»). The model
  learned biblical register and vocabulary; modern conversational input often
  gets a biblical-sounding paraphrase. Psalms and Daniel lections could not be
  used because the modern Bible dataset only contains Genesis, Exodus, Matthew,
  Mark, Luke and John; Ełišē's history is a red link on Wikisource.
* **Size:** ~3.4k pairs is far below what a translation model normally needs.
  Both models memorise phrasing patterns more than they generalise.
* **Evaluation:** one reference per verse; chrF/BLEU reward surface overlap, not
  theological accuracy.
* **Orthography:** training text uses classical orthography (եաւ, յ, զ-), so the
  output keeps it; modern reform spelling input is still understood.

## Data attribution

* Grabar text: *Աստուածաշունչ* (Zohrab, 1805), transcribed on hy.wikisource
  (CC BY-SA); plus the bilingual passages of Abeghyan's *Գրաբարի դասագիրք*
  (1916) on hy.wikisource.
* Modern Armenian text: `k-mktr/armenian_bible_hy` on Hugging Face.
* Base model: `facebook/nllb-200-distilled-600M` (CC BY-NC 4.0), used through a
  LoRA adapter — check that licence before commercial use.

Code: MIT.
