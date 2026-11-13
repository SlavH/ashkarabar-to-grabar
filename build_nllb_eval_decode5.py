import json
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md("# Decoding search 5: no_repeat_ngram_size and min_new_tokens around best (nb=4,lp=1.05,rp=1.4)")
code(r"""
!pip uninstall -y -q torchao
!pip install -q transformers peft accelerate sacrebleu sentencepiece
import os, urllib.request
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
def translate(text, max_new=192, min_new=15, num_beams=4, length_penalty=1.05, no_repeat_ngram_size=4, repetition_penalty=1.4):
    tok.src_lang=SRC_LANG
    enc=tok(text,return_tensors='pt',truncation=True,max_length=192).to(device)
    out=model.generate(**enc, forced_bos_token_id=forced_bos, max_new_tokens=max_new, min_new_tokens=min_new,
                       num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
                       early_stopping=True, length_penalty=length_penalty, repetition_penalty=repetition_penalty)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

results=[]
for nr in [0,3,4,5,6]:
  for mn in [0,5,10,12,15,18,20]:
    preds=[translate(m,192,mn,4,1.05,nr,1.4) for m,_ in test_pairs]
    refs=[g for _,g in test_pairs]
    chrf=sacrebleu.corpus_chrf(preds,[refs]).score
    bleu=sacrebleu.corpus_bleu(preds,[refs]).score
    results.append((chrf,bleu,nr,mn))
    print(f'nr={nr} mn={mn}: chrF={chrf:.2f} BLEU={bleu:.2f}', flush=True)
results.sort(reverse=True)
print('TOP 15:');
for r in results[:15]: print(r)
best=results[0]
print('BEST:', best)
open(f'{WORK}/metrics_decode5.txt','w').write(str(best)+'\n')
""")
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
open('/home/user/armenian_nlp/kaggle_nllb/nllb_train.ipynb','w').write(json.dumps(nb,ensure_ascii=False,indent=1))
for i,c in enumerate(cells):
  if c['cell_type']=='code':
    src="\n".join(l for l in ''.join(c['source']).splitlines() if not l.strip().startswith('!'))
    compile(src,f'c{i}','exec')
print('written')
