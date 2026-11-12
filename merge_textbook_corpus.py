"""Combine the Bible corpus with the textbook corpus into the v3 training set.

Sources (all modern \t grabar, column 0 = source, column 1 = target)
--------------------------------------------------------------------
  data/train_real.tsv     Zohrab 1805 Grabar Bible (Gen, Ex) vs modern
  data/textbook_pairs.tsv «Գրաբարի դասագիրք» gospel passages vs modern Bible
  data/dasagirq_pairs.tsv «Գրաբարի դասագիրք» 1916 modern translations (pre-reform
                          spelling) vs the same passages' grabar

The held-out test split is never touched, so numbers stay comparable with the
v2 model (chrF 51.22 / BLEU 22.22 on the same 108 verses).  Textbook pairs
that would leak into val/test are dropped, not shuffled in.

Output
------
data/train_v3.tsv   Bible train verses + textbook pairs
data/val_v3.tsv     Bible val verses + a small slice of textbook pairs
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def read(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if len(a) > 10 and len(b) > 10:
                out.append((a, b))
    return out


def main():
    bible = read(os.path.join(DATA, 'train_real.tsv'))
    val = read(os.path.join(DATA, 'val_real.tsv'))
    test = read(os.path.join(DATA, 'test_real.tsv'))
    text_t = read(os.path.join(DATA, 'textbook_pairs.tsv'))
    text_d = read(os.path.join(DATA, 'dasagirq_pairs.tsv'))
    print('bible train %d  val %d  test %d  textbook(bible-align) %d  textbook(1916) %d'
          % (len(bible), len(val), len(test), len(text_t), len(text_d)))

    # leakage guard: no verse may appear on both sides of a split boundary
    protected_mod = {a for a, _ in val} | {a for a, _ in test}
    protected_grb = {b for _, b in val} | {b for _, b in test}
    leaked = 0

    def clean(rows):
        nonlocal leaked
        out = []
        for a, b in rows:
            # check for exact duplicates in val/test
            if a in protected_mod or b in protected_grb:
                leaked += 1
                continue
            out.append((a, b))
        return out

    text = clean(text_t + text_d)
    print('textbook pairs after leakage guard: %d (dropped %d)' % (len(text), leaked))

    rng = random.Random(1234)
    rng.shuffle(text)
    n_val = max(40, int(0.05 * len(text)))
    text_val, text_train = text[:n_val], text[n_val:]

    def dump(path, rows):
        seen = set()
        with open(path, 'w', encoding='utf-8') as f:
            for a, b in rows:
                if b in seen:
                    continue
                seen.add(b)
                f.write('%s\t%s\n' % (a, b))
        return len(seen)

    n_train = dump(os.path.join(DATA, 'train_v3.tsv'), bible + text_train)
    n_val = dump(os.path.join(DATA, 'val_v3.tsv'), val + text_val)
    print('train_v3 %d   val_v3 %d   (test untouched: %d)' % (n_train, n_val, len(test)))

    for a, b in text_train[:3]:
        print('\nMOD:', a[:140])
        print('GRB:', b[:140])


if __name__ == '__main__':
    main()