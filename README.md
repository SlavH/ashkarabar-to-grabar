# Armenian MT: Ashkharabar → Grabar

<sup>[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)</sup>
<sup>[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org)</sup>
<sup>[![Kaggle](https://img.shields.io/badge/trained%20on-kaggle-blue.svg)](https://kaggle.com)</sup>
<sup>[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)</sup>

Neural machine translation model that translates from **Modern Armenian (āshkharabar)** to **Classical Armenian (grabar)** — an ancient literary language with very limited NLP resources.

---

## What This Is

A **Transformer Encoder-Decoder** trained from scratch on PyTorch. Given a sentence in modern Armenian, it generates the classical (grabar) equivalent.

**Current status:** Working prototype trained on ~1,000 sentence pairs. The model loads and runs inference — next step is more parallel data and longer training.

---

## Project Structure

```
armenian-mt-grabar/
├── checkpoints/
│   └── armenian_mt.pt          # Trained model weights (32.9 MB)
├── data/
│   ├── train.csv               # 1,000 training pairs
│   ├── val.csv                 # 300 validation pairs
│   └── raw_pairs.txt           # Raw parallel corpus (add your own data here)
├── tokenizer/
│   └── sp_model.json           # BPE tokenizer (4,000 tokens)
├── src/
│   ├── model.py                # Transformer Encoder-Decoder (from scratch)
│   ├── train.py                # Training loop with checkpointing
│   ├── infer.py                # Single-sentence translation
│   ├── prepare_data.py         # Data cleaning & split
│   └── train_tokenizer.py      # BPE tokenizer training
├── test_one.py                 # Translate one sentence
├── test_batch.py               # Test on validation set
├── eval_metrics.py             # chrF & BLEU scores
├── kaggle_train.ipynb          # Kaggle GPU training notebook
├── run_tests.bat               # Windows test runner
├── run_on_D.bat                # Full training pipeline (Windows)
└── README.md
```

---

## Quick Start

### Install dependencies

```bash
pip install torch tokenizers sentencepiece tqdm
```

### Translate a sentence

```bash
python test_one.py --text "Բարև, աշխարհ"
```

### Batch test on validation set

```bash
python test_batch.py 20
```

### Evaluate with chrF & BLEU

```bash
pip install sacrebleu
python eval_metrics.py
```

### Train more (locally)

```bash
python src\train.py
```

### Train on Kaggle GPU (recommended for speed)

Upload `kaggle_train.ipynb` to [kaggle.com/code](https://kaggle.com/code), enable GPU (T4 x2), run all cells, and download the checkpoint.

---

## Why This Matters

Classical Armenian (**grabar**) has a rich literary heritage but very few NLP resources. This project is a step toward making it accessible to modern NLP pipelines — translation, search, and digital preservation.

---

## Roadmap

- [ ] Collect larger parallel corpus (Bible, historical texts)
- [ ] Train longer epochs on Kaggle T4 GPU
- [ ] Improve BLEU/chrF scores
- [ ] Deploy as a simple web demo

---

## Example

```bash
$ python test_one.py --text "Բարև, աշխարհ"
Input  (ASH): Բարև, աշխարհ
Output (GRB): Պարով, աշխարհ
```

---

## License

[MIT](LICENSE)

---

**Kaggle notebook:** [`slavhayrapetyan/armenian-mt-ashkharabar-to-grabar`](https://www.kaggle.com/code/slavhayrapetyan/armenian-mt-ashkharabar-to-grabar)