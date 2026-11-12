"""Local CPU dry-run of the new Kaggle notebook (2 epochs)."""
import json, os, shutil, urllib.request

SIM = '/tmp/kgsim2'
REPO = '/home/user/armenian_nlp'
shutil.rmtree(SIM, ignore_errors=True)
os.makedirs(SIM)

_real = urllib.request.urlretrieve


def fake(url, dest=None):
    rel = url.split('/main/')[-1]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy(os.path.join(REPO, rel), dest)
    return dest, None


urllib.request.urlretrieve = fake

nb = json.load(open(REPO + '/kaggle_train.ipynb'))
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
src = '\n'.join(codes)
src = src.replace('/kaggle/working', SIM)
src = src.replace('EPOCHS = 200', 'EPOCHS = 1')
src = src.replace('if ep % 5 == 0 or ep == 1:', 'if True:')
src = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('!'))
exec(compile(src, 'notebook', 'exec'), {'__name__': 'nb'})
print('\nDRY RUN OK')
