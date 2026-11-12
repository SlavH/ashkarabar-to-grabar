"""Augment training data with light character-level noise on the modern side."""
import os, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

random.seed(42)

def noise_text(text, p=0.02):
    """Apply very light character-level noise (swaps only) to Armenian text."""
    chars = list(text)
    out = list(chars)
    i = 0
    while i < len(out) - 1:
        if random.random() < p and out[i].isalpha() and out[i+1].isalpha() and out[i] != ' ' and out[i+1] != ' ':
            out[i], out[i+1] = out[i+1], out[i]
            i += 2
        else:
            i += 1
    return ''.join(out)

def main():
    pairs = []
    with open(os.path.join(DATA, 'train_v3.tsv'), encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    augmented = []
    for mod, grb in pairs:
        noisy_mod = noise_text(mod, p=0.02)
        if noisy_mod != mod:
            augmented.append((noisy_mod, grb))

    print(f'original: {len(pairs)}, augmented: {len(augmented)}')

    with open(os.path.join(DATA, 'train_v3_aug.tsv'), 'w', encoding='utf-8') as f:
        for mod, grb in pairs + augmented:
            f.write(f'{mod}\t{grb}\n')

    print(f'written: {len(pairs) + len(augmented)} pairs to train_v3_aug.tsv')

if __name__ == '__main__':
    main()
