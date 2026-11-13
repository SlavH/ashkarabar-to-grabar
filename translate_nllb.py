# -*- coding: utf-8 -*-
"""
infer_nllb.py
Modern Armenian (ashkharabar) -> Classical Armenian (grabar) with a fine-tuned
NLLB-200 model.

The repository only stores the small LoRA adapter
(`checkpoints/nllb_lora_grabar/`); the 600M base model is downloaded from
Hugging Face the first time you run this.

    python3 translate_nllb.py --text "Աստուած ստեղծեց երկինքն ու երկիրը։"

Requires: transformers, peft, sentencepiece, torch.
"""

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
ADAPTER_DIR = BASE / "checkpoints" / "nllb_lora_grabar"

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "hye_Armn"   # Modern Armenian
TGT_LANG = "hye_Armn"   # NLLB-200 has no separate grabar / hyw tag
MAX_LEN = 192


def load_model(adapter_dir: Path = ADAPTER_DIR, device: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {adapter_dir}\n"
            "Copy it from the Kaggle output (kaggle_output/nllb_lora_grabar)."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def translate(model, tokenizer, text: str, device: str = "cpu",
              num_beams: int = 4, max_new_tokens: int = MAX_LEN,
              length_penalty: float = 1.1, repetition_penalty: float = 1.3,
              no_repeat_ngram_size: int = 4) -> str:
    import torch

    tokenizer.src_lang = SRC_LANG
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=forced_bos,
                             max_new_tokens=max_new_tokens, num_beams=num_beams,
                             no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=True,
                             length_penalty=length_penalty,
                             repetition_penalty=repetition_penalty)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Translate ashkharabar -> grabar (NLLB)")
    parser.add_argument("--text", required=True)
    parser.add_argument("--adapter", default=str(ADAPTER_DIR))
    parser.add_argument("--beams", type=int, default=4)
    args = parser.parse_args()

    model, tokenizer, device = load_model(Path(args.adapter))
    result = translate(model, tokenizer, args.text, device=device, num_beams=args.beams)

    print(f"\nInput  (ASH): {args.text}")
    print(f"Output (GRB): {result}")


if __name__ == "__main__":
    main()
