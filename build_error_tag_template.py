"""Generate a per-example template for manual/assisted error tagging on the frozen test set.

Usage:
  python3 build_error_tag_template.py [--preds preds.txt] [--out reports/test_error_tags.tsv]

- Reads data/test_real.tsv (ash\tgrabar).
- If --preds is provided (one prediction per line, aligned to test order), includes MODEL column.
- Outputs a TSV with columns for tagging: idx, ash, gold, model, chrf_note, error_type, comment.
"""
import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default=None)
    ap.add_argument('--out', default='reports/test_error_tags_template.tsv')
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base, 'data', 'test_real.tsv')
    pairs = []
    with open(test_path, encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue
            a, b = line.rstrip('\n').split('\t', 1)
            pairs.append((a.strip(), b.strip()))
    preds = [''] * len(pairs)
    if args.preds:
        with open(args.preds, encoding='utf-8') as f:
            preds = [l.rstrip('\n') for l in f]
        if len(preds) < len(pairs):
            preds += [''] * (len(pairs) - len(preds))
        preds = preds[:len(pairs)]
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ['idx', 'ashkharabar', 'gold_grabar', 'model_grabar', 'error_type', 'comment']
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for i, ((a, g), m) in enumerate(zip(pairs, preds), 1):
            row = [str(i), a, g, m, '', '']
            f.write('\t'.join(x.replace('\t', ' ').replace('\n', ' ') for x in row) + '\n')
    print(f'wrote {out_path} rows={len(pairs)} preds={"yes" if args.preds else "no"}')

if __name__ == '__main__':
    main()
