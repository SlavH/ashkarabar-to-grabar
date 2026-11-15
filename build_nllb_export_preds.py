import json
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md("# Export predictions on frozen test (current best decode)")
code(r"""
!pip uninstall -y -q torchao
!pip install -q transformers peft accelerate sacrebleu sentencepiece
import os, json, urllib.request
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch, sacrebleu
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

WORK='/kaggle/working'; RAW='https://raw.githubusercontent.com/SlavH/ashkarabar-to-grabar/main'
os.makedirs(f'{WORK}/data', exist_ok=True); os.makedirs(f'{WORK}/adapter', exist_ok=True)
urllib.request.urlretrieve(f'{RAW}/data/test_real.tsv', f'{WORK}/data/test_real.tsv')
for f in ['adapter_config.json','adapter_model.safetensors']:
    urllib.request.urlretrieve(f'{RAW}/checkpoints/nllb_lora_grabar/{f}', f'{WORK}/adapter/{f}')

def read_pairs(path):
    out=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if '\t' not in line: continue
            a,b=line.rstrip('\n').split('\t',1)
            if len(a.strip())>10 and len(b.strip())>10: out.append((a.strip(),b.strip()))
    return out

test_pairs=read_pairs(f'{WORK}/data/test_real.tsv')
MODEL_NAME='facebook/nllb-200-distilled-600M'; SRC_LANG='hye_Armn'; TGT_LANG='hye_Armn'
tok=AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model=PeftModel.from_pretrained(model,f'{WORK}/adapter')
device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device).eval()
forced_bos=tok.convert_tokens_to_ids(TGT_LANG)

@torch.no_grad()
def translate(text):
    tok.src_lang=SRC_LANG
    enc=tok(text,return_tensors='pt',truncation=True,max_length=192).to(device)
    out=model.generate(**enc, forced_bos_token_id=forced_bos, max_new_tokens=192, min_new_tokens=20,
                       num_beams=4, no_repeat_ngram_size=0,
                       early_stopping=True, length_penalty=1.07, repetition_penalty=1.22)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

rows=[]
preds=[]; refs=[]
for i,(src,ref) in enumerate(test_pairs,1):
    pred=translate(src)
    rows.append({'i':i,'src':src,'pred':pred,'ref':ref})
    preds.append(pred); refs.append(ref)
    if i<=3: print(i, pred[:120], flush=True)
chrf=sacrebleu.corpus_chrf(preds,[refs]).score
bleu=sacrebleu.corpus_bleu(preds,[refs]).score
print('chrF',chrf,'BLEU',bleu, flush=True)
with open(f'{WORK}/preds_test_best.jsonl','w',encoding='utf-8') as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
open(f'{WORK}/metrics_preds.txt','w').write(f'chrF={chrf}\nBLEU={bleu}\n')
""")
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
open('/home/user/armenian_nlp/kaggle_nllb/nllb_train.ipynb','w',encoding='utf-8').write(json.dumps(nb,ensure_ascii=False,indent=1))
for i,c in enumerate(cells):
  if c['cell_type']=='code':
    src="\n".join(l for l in ''.join(c['source']).splitlines() if not l.strip().startswith('!'))
    compile(src,f'c{i}','exec')
print('written')
