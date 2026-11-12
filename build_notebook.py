"""Generate the Kaggle training notebook (clean, runnable)."""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md("""# Ashkharabar → Grabar: Transformer MT on a real parallel corpus

Training data: **2,715 verse-aligned pairs** from the Zohrab 1805 Grabar Bible
(hy.wikisource) matched verse-by-verse with the modern Eastern Armenian Bible
(`k-mktr/armenian_bible_hy`).

The architecture is imported from the repository (`src/model.py`) so the exported
checkpoint loads directly in `test_one.py` / `src/infer.py`.
""")

code("""
!pip install -q tokenizers pandas
import os, sys, time, math, random, urllib.request
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WORK = '/kaggle/working'
RAW = 'https://raw.githubusercontent.com/SlavH/ashkarabar-to-grabar/main'
os.makedirs(f'{WORK}/src', exist_ok=True)
os.makedirs(f'{WORK}/checkpoints', exist_ok=True)
os.makedirs(f'{WORK}/tokenizer', exist_ok=True)
os.makedirs(f'{WORK}/data', exist_ok=True)

for rel in ['src/__init__.py', 'src/model.py', 'data/train_real.tsv', 'data/val_real.tsv']:
    dest = os.path.join(WORK, rel)
    url = f'{RAW}/{rel}'
    urllib.request.urlretrieve(url, dest)
    print(rel, os.path.getsize(dest), 'bytes')

sys.path.insert(0, WORK)
from src.model import TransMT, BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')
print(torch.__version__)
torch.manual_seed(1234); random.seed(1234)
torch.set_float32_matmul_precision('high')
""")

code("""
# ---------------- data ----------------
def read_pairs(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\\n')
            if '\\t' not in line:
                continue
            s, t = line.split('\\t', 1)
            s, t = s.strip(), t.strip()
            if len(s) > 10 and len(t) > 10:
                out.append((s, t))
    return out

train_pairs = read_pairs(f'{WORK}/data/train_real.tsv')
val_pairs = read_pairs(f'{WORK}/data/val_real.tsv')
print('train:', len(train_pairs), 'val:', len(val_pairs))
print('example ASH:', train_pairs[0][0][:100])
print('example GRB:', train_pairs[0][1][:100])
""")

code("""
# ---------------- tokenizer ----------------
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 2400
MAXLEN = 160

tok = Tokenizer(models.BPE(unk_token='[UNK]'))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=['[UNK]', '[BOS]', '[EOS]', '[PAD]'],   # -> 0,1,2,3
    min_frequency=2,
    show_progress=False,
)
tok.train_from_iterator([x for p in train_pairs for x in p], trainer)
VOCAB = tok.get_vocab_size()
tok.save(f'{WORK}/tokenizer/sp_model.json')
print('vocab:', VOCAB)
print('special ids:', {t: tok.token_to_id(t) for t in ['[UNK]', '[BOS]', '[EOS]', '[PAD]']})
print('sample encode:', tok.encode(train_pairs[0][0]).tokens[:12])
""")

code("""
# ---------------- dataset (bidirectional) ----------------
class PairDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        ai = [BOS_IDX] + tok.encode(a).ids[:MAXLEN - 2] + [EOS_IDX]
        bi = [BOS_IDX] + tok.encode(b).ids[:MAXLEN - 2] + [EOS_IDX]
        return torch.tensor(ai), torch.tensor(bi)


def collate(batch):
    src, tgt = zip(*batch)
    return (nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX),
            nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX))


BATCH = 24
dl_train = DataLoader(PairDS(train_pairs), batch_size=BATCH, shuffle=True, collate_fn=collate)
dl_val = DataLoader(PairDS(val_pairs), batch_size=64, shuffle=False, collate_fn=collate)
print('batches per epoch:', len(dl_train))
""")

code("""
# ---------------- model ----------------
model = TransMT(vocab=VOCAB).to(DEVICE)
print('config:', model.config)
print('params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
""")

code("""
# ---------------- training ----------------
crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01)
EPOCHS = 200
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, total_steps=EPOCHS * len(dl_train), pct_start=0.08,
    anneal_strategy='cos')


def batch_loss(s, t):
    logits = model(s, t[:, :-1])
    return crit(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate():
    model.eval()
    total, n = 0.0, 0
    for s, t in dl_val:
        s, t = s.to(DEVICE), t.to(DEVICE)
        # modern -> grabar (the task we care about)
        l = batch_loss(s, t)
        if torch.isfinite(l):
            total += l.item(); n += 1
    model.train()
    return total / max(1, n)


best, best_state, best_ep = float('inf'), None, 0
t0 = time.time()
log = []
for ep in range(1, EPOCHS + 1):
    model.train()
    tot, nb = 0.0, 0
    for s, t in dl_train:
        s, t = s.to(DEVICE), t.to(DEVICE)
        loss = 0.5 * (batch_loss(s, t) + batch_loss(t, s))   # bidirectional
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        tot += loss.item(); nb += 1

    if ep % 5 == 0 or ep == 1:
        vl = evaluate()
        line = 'epoch %3d/%d train=%.4f val=%.4f lr=%.2e %.0fs' % (
            ep, EPOCHS, tot / max(1, nb), vl, sched.get_last_lr()[0], time.time() - t0)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            line += '  *best*'
        print(line, flush=True)
        log.append(line)

if best_state:
    model.load_state_dict(best_state)
print('BEST val loss %.4f at epoch %d' % (best, best_ep))
""")

code("""
# ---------------- export ----------------
torch.save({
    'model': model.state_dict(),
    'vocab': VOCAB,
    'config': model.config,
    'val_loss': best,
    'epoch': best_ep,
}, f'{WORK}/checkpoints/armenian_mt.pt')

model.eval()

def translate(text, max_len=MAXLEN):
    ids = [BOS_IDX] + tok.encode(text).ids[:max_len - 2] + [EOS_IDX]
    src = torch.tensor(ids, device=DEVICE).unsqueeze(0)
    out = model.greedy_decode(src, max_len=max_len, device=DEVICE).tolist()
    if out and out[0] == BOS_IDX:
        out = out[1:]
    if EOS_IDX in out:
        out = out[:out.index(EOS_IDX)]
    return tok.decode(out).strip()

samples = ['Բարեւ, ինչպէս ես։',
           'Աստուած ստեղծեց երկինքն ու երկիրը։',
           'Ես սիրում եմ հայերէնը։',
           'Ժողովուրդը պատասխան տուեց եւ ասաց.']
print('=== held-out verses ===')
lines = []
for m, g in val_pairs[:10]:
    h = translate(m)
    block = f'ASH : {m}\\nGOLD: {g}\\nMODEL: {h}\\n'
    print(block); lines.append(block)

lines.append('=== free sentences ===')
print('=== free sentences ===')
for s in samples:
    h = translate(s)
    print(f'ASH : {s}\\nMODEL: {h}\\n')
    lines.append(f'ASH : {s}\\nMODEL: {h}\\n')

open(f'{WORK}/samples.txt', 'w', encoding='utf-8').write('\\n'.join(lines))
open(f'{WORK}/train_log.txt', 'w', encoding='utf-8').write('\\n'.join(log))

print('--- output files ---')
for root, _, files in os.walk(WORK):
    for fn in files:
        p = os.path.join(root, fn)
        if '/data/' in p and fn.endswith('.parquet'):
            continue
        print('%10d  %s' % (os.path.getsize(p), p))
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open('/home/user/armenian_nlp/kaggle_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "\n".join(l for l in ''.join(c["source"]).splitlines()
                        if not l.strip().startswith("!"))
        compile(src, f"cell{i}", "exec")
print("notebook written, all code cells compile")
