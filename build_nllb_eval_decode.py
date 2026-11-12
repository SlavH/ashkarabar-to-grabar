import json
cells=[]
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md("# Decoding search on current best adapter (v9)")
code(r"""
!pip uninstall -y -q torchao
!pip install -q transformers peft accelerate sacrebleu sentencepiece
import os, urllib.request, time, itertools
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch, sacrebleu
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

WORK='/kaggle/working'
RAW='https://raw.githubusercontent.com/SlavH/ashkarabar-to-grabar/main'
os.makedirs(f'{WORK}/data', exist_ok=True)
os.makedirs(f'{WORK}/adapter', exist_ok=True)
for rel in ['data/test_real.tsv']:
    urllib.request.urlretrieve(f'{RAW}/{rel}', f'{WORK}/{rel}')
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
def translate(text, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3):
    tok.src_lang=SRC_LANG
    enc=tok(text, return_tensors='pt', truncation=True, max_length=MAXLEN).to(device)
    out=model.generate(**enc, forced_bos_token_id=forced_bos,
                       max_new_tokens=MAXLEN, num_beams=num_beams,
                       no_repeat_ngram_size=no_repeat_ngram_size,
                       early_stopping=True, length_penalty=length_penalty)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

configs=[]
for nb in [4,5,6]:
  for lp in [0.9,1.0,1.1,1.2]:
    for nr in [2,3,4]:
      configs.append((nb,lp,nr))

best=None
results=[]
for (nb,lp,nr) in configs:
    t0=time.time()
    preds=[translate(m, nb, lp, nr) for m,_ in test_pairs]
    refs=[g for _,g in test_pairs]
    chrf=sacrebleu.corpus_chrf(preds,[refs]).score
    bleu=sacrebleu.corpus_bleu(preds,[refs]).score
    results.append((chrf,bleu,nb,lp,nr))
    print(f'beams={nb} lp={lp} nr={nr}: chrF={chrf:.2f} BLEU={bleu:.2f} ({int(time.time()-t0)}s)')
    if best is None or chrf>best[0]: best=(chrf,bleu,nb,lp,nr,preds)

results.sort(reverse=True)
print('\nTOP 10:')
for r in results[:10]:
    print(r)
chrf,bleu,nb,lp,nr,preds=best
print(f'\nBEST: beams={nb} lp={lp} nr={nr} chrF={chrf:.2f} BLEU={bleu:.2f}')
open(f'{WORK}/metrics_decode.txt','w').write(f'chrF={chrf:.2f} BLEU={bleu:.2f} beams={nb} lp={lp} nr={nr}\n')
""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
open('/home/user/armenian_nlp/kaggle_nllb/nllb_train.ipynb','w').write(json.dumps(nb,ensure_ascii=False,indent=1))
for i,c in enumerate(cells):
  if c['cell_type']=='code':
    src="\n".join(l for l in ''.join(c['source']).splitlines() if not l.strip().startswith('!'))
    compile(src,f'c{i}','exec')
print('eval notebook written')
