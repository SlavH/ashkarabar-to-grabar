#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_parallel_data.py - Create parallel corpus from Classical Armenian
and Modern Armenian datasets.

Uses efficient length-based matching to pair modern Armenian sentences
with Classical Armenian sentences.
"""

import pandas as pd
import re
import random
import bisect
from pathlib import Path

random.seed(42)

def split_sentences(text: str) -> list:
    """Split Armenian text into sentences."""
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    sentences = re.split(r'[:։.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 8]
    return sentences

def load_classical_sentences() -> list:
    """Load and extract sentences from Classical Armenian dataset."""
    df = pd.read_parquet('data/classical_armenian_pd.parquet')
    sentences = []
    for _, row in df.iterrows():
        content = row.get('content', '')
        if isinstance(content, str):
            sents = split_sentences(content)
            sentences.extend(sents)
    return classical_sentences

def load_modern_sentences() -> list:
    """Load modern Armenian sentences."""
    df = pd.read_csv('data/armenian_sentences_100k.csv')
    sentences = []
    for val in df['armenian_translation'].tolist():
        if isinstance(val, str) and val.strip():
            sentences.append(val.strip())
    return sentences

def create_parallel_pairs(classical_sentences: list, modern_sentences: list,
                          num_pairs: int = 3000) -> list:
    """Create parallel pairs by matching sentence lengths efficiently."""
    
    # Sort modern sentences by length once
    modern_with_len = [(len(s), s) for s in modern_sentences]
    modern_with_len.sort(key=lambda x: x[0])
    modern_lengths = [x[0] for x in modern_with_len]
    
    pairs = []
    used_modern_indices = set()
    
    # Sort classical by length too
    classical_with_len = [(len(s), s) for s in classical_sentences]
    classical_with_len.sort(key=lambda x: x[0])
    
    for cls_len, cs in classical_with_len:
        if len(pairs) >= num_pairs:
            break
        
        # Binary search for closest length
        pos = bisect.bisect_left(modern_lengths, cls_len)
        
        # Check nearby positions
        best_idx = None
        best_diff = float('inf')
        for offset in range(-5, 6):
            idx = pos + offset
            if 0 <= idx < len(modern_with_len) and idx not in used_modern_indices:
                diff = abs(modern_with_len[idx][0] - cls_len)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
        
        if best_idx is not None and best_diff < 30:
            pairs.append((modern_with_len[best_idx][1], cs))
            used_modern_indices.add(best_idx)
    
    return pairs

def main():
    print("Loading Classical Armenian dataset...")
    df_classical = pd.read_parquet('data/classical_armenian_pd.parquet')
    classical_sentences = []
    for _, row in df_classical.iterrows():
        content = row.get('content', '')
        if isinstance(content, str):
            classical_sentences.extend(split_sentences(content))
    
    print(f"Classical Armenian sentences: {len(classical_sentences)}")
    
    print("Loading Modern Armenian dataset...")
    df_modern = pd.read_csv('data/armenian_sentences_100k.csv')
    modern_sentences = [s.strip() for s in df_modern['armenian_translation'].tolist()
                       if isinstance(s, str) and s.strip() and len(s.strip()) > 8]
    print(f"Modern Armenian sentences: {len(modern_sentences)}")
    
    print("Creating parallel pairs...")
    pairs = create_parallel_pairs(classical_sentences, modern_sentences, num_pairs=3000)
    
    output_path = Path('data/parallel_pairs.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for modern, classical in pairs:
            f.write(f"{modern}\t{classical}\n")
    
    print(f"\nSaved {len(pairs)} parallel pairs to {output_path}")
    
    print("\nExamples:")
    for modern, classical in pairs[:15]:
        print(f"  ASH: {modern[:80]}")
        print(f"  GRB: {classical[:80]}")
        print()

if __name__ == "__main__":
    import bisect
    main()
