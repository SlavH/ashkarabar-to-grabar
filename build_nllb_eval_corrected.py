"""Re-evaluate on corrected test_real.tsv (after applying #49/#107 fixes)."""
import json

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md("# Re-evaluation on corrected test_real.tsv (#49/#107 fixes applied)\n\n"
   "Current best adapter (v9) + best decode config (nb=4, lp=1.07, rp=1.22, mn=20, nr=0).")

code(r"""
!pip uninstall -y -q torchao
!pip install -q transformers peft accelerate sacrebleu sentencepiece
import os, urllib.request, time
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch, sacrebleu
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

WORK='/kaggle/working'
RAW='https://raw.githubusercontent.com/SlavH/ashkarabar-to-grabar/main'
os.makedirs(f'{WORK}/data', exist_ok=True)
os.makedirs(f'{WORK}/adapter', exist_ok=True)
urllib.request.urlretrieve(f'{RAW}/data/test_real.tsv', f'{WORK}/data/test_real.tsv')
for f in ['adapter_config.json','adapter_model.safetensors']:
    urllib.request.urlretrieve(f'{RAW}/checkpoints/nllb_lora_grabar/{f}', f'{WORK}/adapter/{f}')
print('downloaded')

def read_pairs(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            line=line.rstrip('\n')
            if '\t' not in line: continue
            a,b=line.split('\t',1)
            a,b=a.strip(),b.strip()
            if len(a)>10 and len(b)>10: out.append((a,b))
    return out

test_pairs=read_pairs(f'{WORK}/data/test_real.tsv')
print('test', len(test_pairs))

MODEL_NAME='facebook/nllb-200-distilled-600M'
SRC_LANG='hye_Armn'; TGT_LANG='hye_Armn'; MAXLEN=192
tok=AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model=PeftModel.from_pretrained(model, f'{WORK}/adapter')
device='cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()
forced_bos=tok.convert_tokens_to_ids(TGT_LANG)

@torch.no_grad()
def translate(text, num_beams=4, length_penalty=1.07, no_repeat_ngram_size=0,
              repetition_penalty=1.22, min_new_tokens=20):
    tok.src_lang=SRC_LANG
    enc=tok(text, return_tensors='pt', truncation=True, max_length=MAXLEN).to(device)
    out=model.generate(**enc, forced_bos_token_id=forced_bos,
                       max_new_tokens=192, min_new_tokens=min_new_tokens,
                       num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
                       early_stopping=True, length_penalty=length_penalty,
                       repetition_penalty=repetition_penalty)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

# Best known config from E25
preds=[translate(m) for m,_ in test_pairs]
refs=[g for _,g in test_pairs]
chrf=sacrebleu.corpus_chrf(preds,[refs]).score
bleu=sacrebleu.corpus_bleu(preds,[refs]).score
print(f'CORRECTED TEST: chrF={chrf:.2f} BLEU={bleu:.2f}')
open(f'{WORK}/metrics_corrected.txt','w').write(f'chrF={chrf:.2f} BLEU={bleu:.2f}\n')

lines=[f'ASH  : {m}\nGOLD : {g}\nMODEL: {p}\n' for (m,g),p in zip(test_pairs,preds)]
open(f'{WORK}/samples_corrected.txt','w',encoding='utf-8').write('\n'.join(lines))
print('DONE')
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open('/home/user/armenian_nlp/kaggle_nllb/nllb_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
for i, c in enumerate(cells):
    if c['cell_type'] == 'code':
        src = "\n".join(l for l in ''.join(c['source']).splitlines() if not l.strip().startswith('!'))
        compile(src, f'c{i}', 'exec')
print('eval notebook written')