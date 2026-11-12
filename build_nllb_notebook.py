"""Generate the Kaggle notebook that fine-tunes NLLB-200 (LoRA) on the real Grabar corpus."""
import json

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md("""# Ashkharabar → Grabar: NLLB-200 fine-tune (LoRA)

Fine-tunes `facebook/nllb-200-distilled-600M` on **2,715 verse-aligned pairs**
(Zohrab 1805 Grabar Bible ↔ modern Eastern Armenian Bible).

A LoRA adapter is exported (~10 MB) so the result fits in a Git repo; the base
model is pulled from Hugging Face at inference time.

Both source and target use the `hye_Armn` tag: NLLB-200 has no separate
Western-Armenian or Classical-Armenian token, so the model learns the
modern→grabar mapping from the data itself.
""")

code("""
# Kaggle ships an old torchao that makes peft's dispatcher raise ImportError.
!pip uninstall -y -q torchao
!pip install -q transformers peft accelerate sacrebleu sentencepiece
import os
# one T4 keeps memory predictable (NLLB logits are 256k-wide) and avoids
# DataParallel gather OOMs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json, random, time, urllib.request
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
print('transformers', transformers.__version__, 'torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')

WORK = '/kaggle/working'
RAW = 'https://raw.githubusercontent.com/SlavH/ashkarabar-to-grabar/main'
os.makedirs(f'{WORK}/data', exist_ok=True)
for rel in ['data/train_real.tsv', 'data/val_real.tsv', 'data/test_real.tsv']:
    dest = os.path.join(WORK, rel)
    urllib.request.urlretrieve(f'{RAW}/{rel}', dest)
    print(rel, os.path.getsize(dest))
""")

code("""
def read_pairs(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\\n')
            if '\\t' not in line:
                continue
            a, b = line.split('\\t', 1)
            a, b = a.strip(), b.strip()
            if len(a) > 10 and len(b) > 10:
                out.append((a, b))
    return out

train_pairs = read_pairs(f'{WORK}/data/train_real.tsv')
val_pairs = read_pairs(f'{WORK}/data/val_real.tsv')
print('train', len(train_pairs), 'val', len(val_pairs))
print('ASH:', train_pairs[0][0][:100])
print('GRB:', train_pairs[0][1][:100])
""")

code("""
MODEL_NAME = 'facebook/nllb-200-distilled-600M'
SRC_LANG = 'hye_Armn'
TGT_LANG = 'hye_Armn'   # NLLB-200 has no hyw_Armn / grabar token
MAXLEN = 144

tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)  # fp32 + AMP for stability

from peft import LoraConfig, get_peft_model
lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    task_type='SEQ_2_SEQ_LM')
model = get_peft_model(model, lora)
model.print_trainable_parameters()
print('params total %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6))
""")

code("""
class BibleDS(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        tok.src_lang = SRC_LANG
        enc = tok(src, max_length=MAXLEN, truncation=True)
        tok.tgt_lang = TGT_LANG
        lab = tok(text_target=tgt, max_length=MAXLEN, truncation=True)
        return {'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'labels': lab['input_ids']}

ds_train = BibleDS(train_pairs)
ds_val = BibleDS(val_pairs)
print('train items', len(ds_train), 'val items', len(ds_val))
print('sample label tokens', tok.decode(ds_train[0]['labels'][:12]))
""")

code("""
collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, label_pad_token_id=-100)

BATCH = 8
GRAD_ACC = 2
EPOCHS = 30

base_kwargs = dict(
    output_dir=f'{WORK}/nllb_ckpt',
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=3e-4,
    warmup_steps=100,
    num_train_epochs=EPOCHS,
    save_strategy='epoch',
    save_total_limit=2,
    logging_steps=25,
    predict_with_generate=False,
    fp16=True,
    optim='adafactor',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to='none',
    seed=1234,
)

# the eval-interval argument was renamed across transformers versions
args = None
for key in ('eval_strategy', 'evaluation_strategy'):
    try:
        args = Seq2SeqTrainingArguments(**base_kwargs, **{key: 'epoch'})
        print('using', key)
        break
    except TypeError as e:
        print('retry without', key, '->', e)
if args is None:
    args = Seq2SeqTrainingArguments(**base_kwargs)
print('args ready')

trainer_kwargs = dict(model=model, args=args,
                      train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=collator)
try:
    trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=tok)
except TypeError:
    trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=tok)

t0 = time.time()
trainer.train()
print('training seconds', int(time.time() - t0))
print('best checkpoint', trainer.state.best_model_checkpoint, 'loss', trainer.state.best_metric)
""")

code("""
# ---- save adapter + evaluate on held-out verses ----
ADAPTER = f'{WORK}/nllb_lora_grabar'
model.save_pretrained(ADAPTER)
tok.save_pretrained(f'{WORK}/nllb_tokenizer')
print('adapter files:', os.listdir(ADAPTER))
for f_ in os.listdir(ADAPTER):
    print(f_, os.path.getsize(os.path.join(ADAPTER, f_)))

model.eval()
forced_bos = tok.convert_tokens_to_ids(TGT_LANG)

@torch.no_grad()
def translate(text, max_new=MAXLEN):
    tok.src_lang = SRC_LANG
    enc = tok(text, return_tensors='pt', truncation=True, max_length=MAXLEN).to(model.device)
    out = model.generate(**enc, forced_bos_token_id=forced_bos,
                         max_new_tokens=max_new, num_beams=4,
                         no_repeat_ngram_size=3, early_stopping=True)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

test_pairs = read_pairs(f'{WORK}/data/test_real.tsv')
print('test pairs', len(test_pairs))

lines, preds, refs = [], [], []
t0 = time.time()
for i, (m, g) in enumerate(test_pairs):
    hyp = translate(m)
    preds.append(hyp); refs.append(g)
    block = f'ASH  : {m}\\nGOLD : {g}\\nMODEL: {hyp}\\n'
    lines.append(block)
    if i < 12:
        print(block)
print('generation seconds', int(time.time() - t0))

try:
    import sacrebleu
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    summary = f'chrF={chrf:.2f} BLEU={bleu:.2f} n={len(preds)} split=test'
except Exception as e:
    summary = f'metric error {e}'
print('METRICS', summary)

free = ['Բարեւ, ինչպէս ես։',
        'Աստուած ստեղծեց երկինքն ու երկիրը։',
        'Ես սիրում եմ հայերէնը։',
        'Ժողովուրդը պատասխան տուեց եւ ասաց.']
print('=== free sentences ===')
for s in free:
    h = translate(s)
    print(f'ASH  : {s}\\nMODEL: {h}\\n')
    lines.append(f'ASH  : {s}\\nMODEL: {h}\\n')

open(f'{WORK}/samples_nllb.txt', 'w', encoding='utf-8').write('\\n'.join(lines))
open(f'{WORK}/metrics_nllb.txt', 'w', encoding='utf-8').write(summary + '\\n')
print('DONE')
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with open('/home/user/armenian_nlp/kaggle_nllb/nllb_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "\n".join(l for l in ''.join(c["source"]).splitlines()
                        if not l.strip().startswith("!"))
        compile(src, f"cell{i}", "exec")
print("nllb notebook written, all code cells compile")
