#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_one.py - translate one sentence: modern Armenian -> Classical Armenian (grabar)

Usage:
    python3 test_one.py --text "Բարեւ, աշխարհ"
    python3 test_one.py --text "Ես սիրում եմ հայերենը" --checkpoint checkpoints/armenian_mt.pt
"""

import argparse

from src.infer import load_model, translate


def main():
    parser = argparse.ArgumentParser(description="Translate Armenian (ashkharabar -> grabar)")
    parser.add_argument("--text", required=True, help="text in modern Armenian")
    parser.add_argument("--checkpoint", default=None, help="path to model checkpoint")
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    result = translate(model, tokenizer, args.text)

    print(f"\nInput  (ASH): {args.text}")
    print(f"Output (GRB): {result}")


if __name__ == "__main__":
    main()
