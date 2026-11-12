"""Build a real Modern Armenian -> Classical Armenian (Grabar) parallel corpus.

Sources
-------
Grabar side:  hy.wikisource  «Աստուածաշունչ (1805)» — Zohrab's 1805 Grabar Bible,
              verse-tagged with {{տուն|գլուխ=C|տուն=V}} markers.
Modern side:  HF dataset k-mktr/armenian_bible_hy (Armenian Bible, Eastern),
              verse-aligned via verse_id (Gen_1_1 ...).

Output: data/parallel_real.tsv  (modern \t grabar)
"""
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
WS = os.path.join(DATA, 'wikisource')
os.makedirs(WS, exist_ok=True)
UA = 'ArmenianMTResearch/1.0 (open-source MT dataset building)'


def fetch_raw(title, cache):
    path = os.path.join(WS, cache)
    if os.path.exists(path) and os.path.getsize(path) > 2000:
        return open(path, encoding='utf-8').read()
    url = 'https://hy.wikisource.org/w/index.php?' + urllib.parse.urlencode(
        {'title': title, 'action': 'raw'})
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            txt = urllib.request.urlopen(req, timeout=60).read().decode('utf-8')
            if len(txt) > 1000:
                open(path, 'w', encoding='utf-8').write(txt)
                return txt
            print('  short response, retry', attempt)
        except Exception as e:
            print('  fetch error:', e)
        time.sleep(3 + attempt * 3)
    return ''


# ---------------------------------------------------------------- cleaning
TEMPLATE = re.compile(r'\{\{[^{}]*\}\}')
REF = re.compile(r'<ref[^>]*>.*?</ref>', re.S)
REF2 = re.compile(r'<ref[^>]*/>')
TAG = re.compile(r'<[^>]+>')
LINK = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]')
EMPH = re.compile(r"'{2,5}")

# grabar typography: ' (U+0027) is a breathing/stress mark; ՚ likewise
BREATH = re.compile(r"['\u055a\u2019]")


def clean(text):
    text = REF.sub(' ', text)
    text = REF2.sub(' ', text)
    text = TEMPLATE.sub(' ', text)
    text = LINK.sub(r'\1', text)
    text = TAG.sub(' ', text)
    text = EMPH.sub('', text)
    text = BREATH.sub('', text)
    text = text.replace('և', 'եւ').replace('Ւ', 'Ւ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


VERSE = re.compile(
    r'\{\{\s*տուն\s*\|\s*(?:գլուխ\s*=\s*(\d*)\s*\|\s*)?տուն\s*=\s*(\d+)\s*\}\}')
CHAPTER = re.compile(r'==\s*Գլուխ\s*(\d+)\s*==')


def parse_verses(wikitext):
    """Return {(chapter, verse): text} for a wikisource book page.

    Verse numbers restart in every chapter, and most {{տուն}} templates leave
    the chapter parameter empty, so chapter tracking must be positional.
    """
    events = []
    for m in CHAPTER.finditer(wikitext):
        events.append((m.start(), m.end(), 'ch', int(m.group(1)), 0))
    for m in VERSE.finditer(wikitext):
        events.append((m.start(), m.end(), 'vs',
                       int(m.group(1)) if m.group(1) else 0, int(m.group(2))))
    events.sort(key=lambda x: x[0])

    out = {}
    cur_ch = None
    for i, (s, e, kind, ch, vs) in enumerate(events):
        if kind == 'ch':
            cur_ch = ch
            continue
        nxt = events[i + 1][0] if i + 1 < len(events) else len(wikitext)
        t = clean(wikitext[e:nxt])
        if not t:
            continue
        chapter = ch if ch else (cur_ch or 1)
        out.setdefault((chapter, vs), t)
    return out


def main():
    # ---------------------------------------------------------- grabar books
    books = [
        # (wikisource page, HF book name, HF book_number)
        ('Աստուածաշունչ (1805)/Ծնունդք', 'Genesis', 1),
        ('Աստուածաշունչ (1805)/Ելք', 'Exodus', 2),
    ]

    modern = pd.read_parquet(os.path.join(DATA, 'bible_modern_hy.parquet'))
    modern['verse_number'] = modern['verse_number'].astype(int)

    rows = []
    for page, book, bnum in books:
        print('fetching', page)
        wt = fetch_raw(page, book.lower() + '.grabar.txt')
        if not wt:
            print('  FAILED'); continue
        verses = parse_verses(wt)
        print('  parsed grabar verses:', len(verses))

        sub = modern[modern.book_number == bnum]
        print('  modern verses:', len(sub))
        hit = 0
        for _, r in sub.iterrows():
            key = (int(r.chapter), int(r.verse_number))
            g = verses.get(key)
            if not g or len(g) < 8:
                continue
            mod = ' '.join(str(r.text).split())
            if len(mod) < 8:
                continue
            rows.append((mod, g, book, key[0], key[1]))
            hit += 1
        print('  aligned pairs:', hit)

    df = pd.DataFrame(rows, columns=['modern', 'grabar', 'book', 'chapter', 'verse'])
    df = df.drop_duplicates(subset=['grabar'])
    out = os.path.join(DATA, 'parallel_real.tsv')
    df[['modern', 'grabar']].to_csv(out, sep='\t', index=False, header=False, encoding='utf-8')
    print('\nTOTAL PAIRS:', len(df))
    print('mean chars modern: %.1f  grabar: %.1f'
          % (df.modern.str.len().mean(), df.grabar.str.len().mean()))
    print('\n--- samples ---')
    for _, r in df.sample(5, random_state=1).iterrows():
        print('MOD:', r.modern[:150])
        print('GRB:', r.grabar[:150])
        print()
    print('written:', out)


if __name__ == '__main__':
    main()
