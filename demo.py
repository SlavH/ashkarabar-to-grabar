#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive demo: Modern Armenian -> Classical Armenian (Grabar)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_nllb import load_model, translate

BANNER = """
Ashkharabar -> Grabar  (NLLB-200 + LoRA)
Type modern Armenian text, press Enter. Empty line to quit.
"""


def main():
    print(BANNER)
    model, tok, device = load_model()
    while True:
        try:
            line = input("ASH > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        out = translate(model, tok, line, device=device)
        print(f"GRB > {out}\n")


if __name__ == "__main__":
    main()
