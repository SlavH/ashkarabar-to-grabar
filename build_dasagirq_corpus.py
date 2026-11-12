"""Turn the fetched «Գրաբարի դասագիրք» bilingual pages into training pairs.

Both sides of every passage are human productions of the same text: the
Grabar original and a 1916 Modern Armenian translation.  Verses carry
``<sup>N</sup>`` markers on both sides, so most passages align 1:1; the rest
are aligned with a length-based dynamic program.

Output: data/dasagirq_pairs.tsv   (modern \t grabar \t passage \t unit)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data', 'wikisource', 'dasagirq')
OUT = os.path.join(HERE, 'data', 'dasagirq_pairs.tsv')

from fetch_dasagirq import pairs, cache_path

REF = re.compile(r'<ref[^>]*>.*?</ref>', re.S)
REF2 = re.compile(r'<ref[^>]*/>')
TAG = re.compile(r'<[^>]+>')
LINK = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]')
EMPH = re.compile(r"'{2,5}")
TEMPLATE = re.compile(r'\{\{[^{}]*\}\}', re.S)
SUP = re.compile(r'<sup>\s*(\d+)\s*</sup>')
BREATH = re.compile(r"['\u055a\u2019]")
SPACE = re.compile(r'\s+')
CENTER = re.compile(r'<center>.*?</center>\s*', re.S)


def clean(text):
    text = REF.sub(' ', text)
    text = REF2.sub(' ', text)
    text = TEMPLATE.sub(' ', text)
    text = LINK.sub(r'\1', text)
    text = SUP.sub(' ', text)
    text = TAG.sub(' ', text)
    text = re.sub(r'[A-Za-z]', '', text)  # stray Latin letters are editor typos
    text = EMPH.sub('', text)
    text = BREATH.sub('', text)
    text = text.replace('և', 'եւ')
    text = text.replace('\u200b', ' ')
    return SPACE.sub(' ', text).strip()


def body(wikitext):
    m = re.search(r'<div class=prose>(.*?)</div>', wikitext, re.S)
    if m:
        return CENTER.sub(' ', m.group(1))
    m = re.search(r'<poem>(.*?)</poem>', wikitext, re.S)
    if m:
        return CENTER.sub(' ', m.group(1))
    # drop the {{Վերնագիր ...}} header block, keep the rest
    text = TEMPLATE.sub('', wikitext, count=1)
    return CENTER.sub(' ', text)


def units(wikitext):
    """Split a page into (numbered, [text]) — verses when markers exist."""
    raw = body(wikitext)
    marks = list(SUP.finditer(raw))
    if len(marks) >= 3:
        out = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            t = clean(raw[m.end():end])
            if t:
                out.append((int(m.group(1)), t))
        return 'verse', out

    # poem / undivided prose: blank-line blocks, then single lines
    blocks = [clean(b) for b in re.split(r'\n\s*\n', raw)]
    blocks = [b for b in blocks if len(b) > 1]
    if len(blocks) >= 2:
        return 'block', [(i, b) for i, b in enumerate(blocks)]
    lines = [clean(l) for l in raw.splitlines()]
    lines = [l for l in lines if len(l) > 1]
    return 'line', [(i, l) for i, l in enumerate(lines)]


def dp_align(src, tgt):
    """Gale-Church style alignment on relative character lengths."""
    n, m = len(src), len(tgt)
    ls = [len(s) for s in src]
    lt = [len(t) for t in tgt]
    asum, bsum = sum(ls) or 1, sum(lt) or 1
    cov = []
    for i, a in enumerate(ls):
        cov.append([abs(a / asum - b / bsum) for b in lt])

    INF = float('inf')
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    moves = [(1, 1, 1.0), (1, 2, 1.1), (2, 1, 1.1), (1, 0, 1.2), (0, 1, 1.2)]
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            for di, dj, pen in moves:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                if di and dj:
                    add = cov[i][j]
                else:
                    add = 0.25
                nc = c + add * pen
                if nc < cost[ni][nj]:
                    cost[ni][nj] = nc
                    back[ni][nj] = (i, j)

    out = []
    i, j = n, m
    while i or j:
        pi, pj = back[i][j]
        a = ' '.join(src[pi:i])
        b = ' '.join(tgt[pj:j])
        if a and b:
            out.append((a, b))
        i, j = pi, pj
    out.reverse()
    return out


def ratio_ok(a, b):
    if len(a) < 20 or len(b) < 20:
        return False
    r = len(a) / max(len(b), 1)
    return 0.35 <= r <= 2.9


def main():
    if not os.path.isdir(CACHE):
        sys.exit('run fetch_dasagirq.py first')

    rows = []
    for g, m, section, num in pairs():
        gp, mp = cache_path(section, num, 'grabar'), cache_path(section, num, 'modern')
        if not (os.path.exists(gp) and os.path.exists(mp)):
            print('missing pages for', section, num)
            continue
        gtxt, mtxt = open(gp, encoding='utf-8').read(), open(mp, encoding='utf-8').read()
        gkind, gunits = units(gtxt)
        mkind, munits = units(mtxt)

        if gkind == mkind == 'verse' and len(gunits) == len(munits):
            aligned = [(a[1], b[1]) for a, b in zip(gunits, munits)]
            how = '1:1 verses'
        elif len(gunits) == len(munits):
            aligned = [(a[1], b[1]) for a, b in zip(gunits, munits)]
            how = '1:1 ' + gkind
        else:
            aligned = dp_align([u[1] for u in gunits], [u[1] for u in munits])
            how = 'dp %d/%d' % (len(gunits), len(munits))

        kept = [(a, b) for a, b in aligned if ratio_ok(a, b)]
        for a, b in kept:
            rows.append((a, b, '%s-%02d' % (section, num), len(rows)))
        print('%-8s %02d  %-16s %3d -> %3d kept  (%d / %d units)'
              % (section, num, how, len(aligned), len(kept), len(gunits), len(munits)))

    # rows are (modern, grabar): the modern side is 1916 "ashkharhavar"
    # (pre-reform spelling: պիտի, յ-), grabar is the target of the model
    seen, uniq = set(), []
    for mod, grb, s, _ in [(b, a, s, i) for a, b, s, i in rows]:
        if grb in seen:
            continue
        seen.add(grb)
        uniq.append((mod, grb, s))

    with open(OUT, 'w', encoding='utf-8') as f:
        for mod, grb, s in uniq:
            f.write('%s\t%s\t%s\n' % (mod, grb, s))

    chars = sum(len(a) + len(b) for a, b, _ in uniq)
    print('\npairs: %d   chars: %d   mean modern %.0f / grabar %.0f'
          % (len(uniq), chars, sum(len(a) for a, _, _ in uniq) / max(len(uniq), 1),
             sum(len(b) for _, b, _ in uniq) / max(len(uniq), 1)))
    print('written:', OUT)
    for a, b, _ in uniq[:3]:
        print('\nMOD:', a[:160])
        print('GRB:', b[:160])


if __name__ == '__main__':
    main()
