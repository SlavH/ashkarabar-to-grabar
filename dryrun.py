"""Local CPU dry-run of the Kaggle notebook cells (tiny, 3 epochs)."""
import json, os, shutil, sys

SIM = '/tmp/kgsim'
shutil.rmtree(SIM, ignore_errors=True)
os.makedirs(SIM + '/data', exist_ok=True)
os.makedirs(SIM + '/checkpoints', exist_ok=True)

# reuse already-downloaded corpora
for f in ('classical_armenian_pd.parquet', 'armenian_sentences_100k.csv'):
    src = '/home/user/armenian_nlp/data/' + f
    if os.path.exists(src):
        shutil.copy(src, SIM + '/data/' + f)
    else:
        print('MISSING', src); sys.exit(1)

nb = json.load(open('/home/user/armenian_nlp/kaggle_train.ipynb'))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
print('code cells:', len(codes))

src = '\n'.join(codes[0:])          # imports -> dataset build -> tokenizer -> model -> train -> save
src = src.replace('/kaggle/working', SIM)
src = src.replace('EPOCHS = 120', 'EPOCHS = 3').replace('EPOCHS=120', 'EPOCHS=3')
src = src.replace("if ep % 5 == 0 or ep == 1:", "if True:")
src = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('!'))
exec(compile(src, 'notebook', 'exec'), {'__name__': 'nb'})
print('\nDRY RUN OK')
