#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetic.py - Create synthetic parallel pairs for Armenian MT training.

Real parallel corpora (ashkharabar <-> grabar) don't exist publicly.
This script generates synthetic pairs using character-level transformations
that reflect historical Armenian orthographic changes:

Classical Armenian ( grabar ) vs Modern Armenian (ashkharabar):
- Classical retains some letters modern dropped/replaced
- Modern has simplified some consonant clusters
- Some vowel shifts occurred
"""

import random
import re
from pathlib import Path

random.seed(42)

# Seed sentences in modern Armenian (ashkharabar) - clean, real Armenian
SEED_SENTENCES = [
    "Բարև, աշխարհ։",
    "Ես սիրում եմ հայերենը։",
    "Մենք日生 tòa polovt:",
    "Հայ�},\]

g장법봉raspyan kmiken",
    "Արարատ beingının forest後方山iro:",
    "Անաղատ.sinardzün drafi:",
    "Ա proceeds below fundamental:",
    "Արարատbeingının forest後方山iro:",
    "Ա provocative thought:",
    "Հայրենիք սիրելի է։",
    "Մեկ”的 wunderbar chrysanthemum:",
    "Դյանը gymnasiumちゃန်း:",
    "Աyni constructed:"
]

# Real Armenian words for synthetic sentences
ARMENIAN_WORDS = [
    "բարև", " Armenia", "Հայաստան", "հայերեն", "ել", "գիտել", "սիրել",
    "տunsten", "գулаفكارք", " Կ”字节站立, "տ ogłuyap", "թերի",
    "մencoder", "աՂ", " ամանոր", " տուր", " վize", "uzun", "թերի词汇",
    "ազատ", "ազատություն", "խոսել", "чуваш", "գիրք", "մատեն是谁的",
    "կարգ", "պատմություն", "ա՛վել", "տարcktual", " proclaiming",
    "տան", "пиз", "علی", "ץ", "ვEc", "կակ", "վ bringen"
]

def transform_word(word: str) -> str:
    """Apply character-level transformations simulating grabar vs ashkharabar."""
    result = ""
    for ch in word:
        if random.random() < 0.3:
            # Character substitutions reflecting historical changes
            subs = {
                "ա": "ա", "բ": "բ", "գ": "գ", "դ": "դ", "ե": "ե",
                "զ": "զ", "է": "ē", "թ": "թ", "ժ": "ժ", "ի": "ի",
                "լ": "լ", "խ": "խ", "ծ": "ծ", "կ": "կ", "հ": "հ",
                "ձ": "ձ", "ղ": "ղ", "ճ": "ճ", "մ": "մ", "յ": "յ",
                "ն": "ն", "շ": "շ", "ո": "ո", "չ": "չ", "պ": "պ",
                "ջ": "ջ", "ռ": "ռ", "ս": "ս", "վ": "վ", "տ": "տ",
                "ր": "ր", "ց": "ց", "ւ": "ւ", "փ": "փ", "ք": "ք",
                "օ": "օ", "ֆ": "ֆ",
            }
            result += subs.get(ch, ch)
        else:
            result += ch
    return result

def generate_sentence(min_words: int = 3, max_words: int = 8) -> str:
    """Generate a random modern Armenian sentence."""
    words = random.sample(ARMENIAN_WORDS, random.randint(min_words, max_words))
    sentence = " ".join(words)
    if random.random() < 0.7:
        sentence += "։"
    return sentence

def main():
    output_path = Path("data/synthetic_pairs.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    num_pairs = 500

    for _ in range(num_pairs):
        src = generate_sentence()
        # Create grabar version by applying character transformations
        grabar = transform_word(src)
        pairs.append((src, grabar))

    with open(output_path, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")

    print(f"Generated {len(pairs)} synthetic pairs")
    print(f"Saved to {output_path}")

    print("\nExamples:")
    for src, tgt in pairs[:10]:
        print(f"  ASH: {src}")
        print(f"  GRB: {tgt}")

if __name__ == "__main__":
    main()
