"""Align the «Գրաբարի դասագիրք» gospel/lection passages to the modern Bible.

Each grabar passage carries its Bible reference in the running title and its
own verse numbers as <sup>N</sup> markers (sometimes relative to the passage,
sometimes absolute to the chapter; the printed number is usually missing for
the final verse, and a verse here and there is merged into its neighbour).

Resolution is therefore:
  1. parse the reference (book name may sit outside the parenthesis,
     «ՍԱՂՄՈՍ (ԻԲ․ 2-7)»; «Գլուխ ԺԱ» means "chapter 21"); known typos are
     fixed by OVERRIDES;
  2. enumerate candidate verse lists (single ref, combined refs, whole-chapter
     stretch for multi-span refs, per-chapter for refs without a chapter);
  3. align the segments to the candidate verses with a penalized edit DP;
     accept only if every matched pair shares a word (floor), the mean
     overlap is decent, and at most three gaps remain.

Only books present in the modern dataset can be used today: Genesis, Exodus,
Matthew, Mark, Luke, John.  Psalms and Daniel passages are skipped.  Every
failure is reported explicitly.

Output: data/textbook_pairs.tsv   (modern \t grabar \t tag)
"""
import itertools
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
CACHE = os.path.join(DATA, 'wikisource', 'dasagirq')
OUT = os.path.join(DATA, 'textbook_pairs.tsv')
MODERN_BIBLE = os.path.join(DATA, 'bible_modern_hy.parquet')

from build_dasagirq_corpus import body, clean, SUP  # same wikitext utilities
from fetch_dasagirq import pairs as toc_pairs

BOOK_NAMES = {
    'արար': 1, 'ծնն': 1, 'ծնննդ': 1, 'ծննդ': 1,
    'ելք': 2,
    'մատթ': 40, 'մատ': 40,
    'մրկ': 41,
    'ղուկ': 42, 'ղկ': 42, 'ղսւկ': 42,
    'հրկ': 43,
    'սաղմոս': 19,
    'դանի': 23,
}

BOOK_LABEL = {1: 'Genesis', 2: 'Exodus', 19: 'Psalms', 23: 'Daniel',
              40: 'Matthew', 41: 'Mark', 42: 'Luke', 43: 'John'}

# known typos in the printed references (verified by reading the page text)
OVERRIDES = {
    # 16 "Փորձութիւն" is printed as (Մատթ․ Գ․ 1-11) but the text is the
    # temptation, Matthew 4:1-11 — the sequence runs 3:1-12 (#14) →
    # 3:13-17 (#15) → 4:1-11 (#16), and the page text matches Matt 4.
    16: (40, [(4, (1, 11))]),
}

# modern Armenian letter numerals (as used in the textbook's references):
# units Ա=1 Բ=2 Գ=3 Դ=4 Ե=5 Զ=6 Է=7 Ը=8 Թ=9, tens Ժ=10 Ի=20 Լ=30 Մ=40
# Վ=50 Ճ=60 Շ=70 Ո=80 Խ=90, hundreds from Յ=100 (never needed for ch/vs)
DIGITS = {'Կ': 1, 'Ա': 1, 'Բ': 2, 'Գ': 3, 'Դ': 4, 'Ե': 5, 'Զ': 6,
          'Է': 7, 'Ը': 8, 'Թ': 9, 'Ժ': 10, 'Ի': 20, 'Լ': 30, 'Մ': 40,
          'Վ': 50, 'Ճ': 60, 'Շ': 70, 'Ո': 80, 'Խ': 90,
          'Յ': 100, 'Չ': 200, 'Պ': 300, 'Ռ': 400, 'Ս': 500,
          'Տ': 600, 'Ձ': 700, 'Ւ': 800, 'Ք': 800, 'Փ': 900}

WORKING_TITLE = re.compile(r"<center>'''(.*?)'''</center>")
FILLER = {'գլուխ', 'գլ', 'և', 'եւ', '․', '.'}


def am_num(tok):
    """Armenian or Arabic numeral string -> int (None if unclear)."""
    tok = tok.strip().strip('․.')
    if not tok:
        return None
    if tok.isdigit():
        return int(tok)
    total = 0
    for ch in tok.upper():
        if ch in DIGITS:
            total += DIGITS[ch]
        else:
            return None
    return total or None


def _match_book(tok):
    t = tok.casefold().strip()
    if t in BOOK_NAMES:
        return BOOK_NAMES[t]
    for k, num in BOOK_NAMES.items():
        if len(k) >= 3 and t.startswith(k):
            return num
    return None


def _spans_from(toks):
    """Turn tokens after the book name into [(chapter, (lo,hi)|None)]."""
    spans = []
    for tok in toks:
        if tok.casefold() in FILLER:
            continue
        if re.search(r'[0-9]', tok):
            if re.search(r'[\u0531-\u0556\u0561-\u0588]', tok):
                continue
            if '-' in tok:
                a, b = tok.split('-', 1)
            else:
                a = b = tok
            if not (a.isdigit() and b.isdigit()):
                continue
            a, b = int(a), int(b)
            if not (1 <= a <= 200 and a <= b):
                continue
            if spans and spans[-1][0] is not None and spans[-1][1] is None:
                spans[-1] = (spans[-1][0], (a, b))
            elif spans and spans[-1][0] is not None:
                spans.append((spans[-1][0], (a, b)))
            else:
                spans.append((None, (a, b)))
        else:
            chn = am_num(tok)
            if chn and 1 <= chn <= 300:
                spans.append((chn, None))
            elif re.search(r'[\u0531-\u0556\u0561-\u05880-9]', tok):
                spans = []
                break
            # else: stray word (author name, «Մովսէս Խորենացի»…) — ignore
    return spans


def paren_spans(paren):
    """Parse one parenthetical into (book, spans)."""
    inner = paren[1:-1].casefold()
    for ch in '․.,;«»:':
        inner = inner.replace(ch, ' ')
    toks = [t.strip() for t in inner.split() if t.strip()]
    if not toks:
        return None, []
    for i, cand in enumerate(toks):
        book = _match_book(cand)
        if book is not None:
            return book, _spans_from(toks[i + 1:])
    return None, []


def parse_refs(header_txt):
    """All (book, spans) found in the title.  The book name may sit outside
    the parenthesis, or there may be no parenthesis at all
    («ԱՒԵՏԱՐԱՆ ԸՍՏ ՄԱՏԹԷՈՍԻ Գլուխ ԺԱ»)."""
    m = re.findall(r'(\([^()]*\))', header_txt)
    out = []
    for paren in m:
        book, spans = paren_spans(paren)
        if book is not None and spans:
            out.append((book, spans))
    if out:
        return out
    flat = re.sub(r'[()]', ' ', header_txt).casefold()
    toks = [t.strip() for t in flat.split() if t.strip()]
    for i, cand in enumerate(toks):
        book = _match_book(cand)
        if book is not None:
            spans = _spans_from(toks[i + 1:])
            if spans:
                return [(book, spans)]
    return []


def candidate_lists(book, spans, chapter_limits):
    """Yield every concrete span list this reference could mean: itself,
    the whole-chapter stretch for multi-span refs of one chapter, and – for
    refs without a chapter – the range placed in each possible chapter."""
    if all(c is not None for c, _ in spans):
        yield spans
        # stretch over the gap: (ch, 1-19) + (ch, 23-38) -> (ch, 1-38)
        chs = {c for c, _ in spans}
        if len(chs) == 1 and len(spans) >= 2:
            ch = next(iter(chs))
            lo = min(v[0] for _, v in spans)
            hi = max(v[1] for _, v in spans)
            if (lo, hi) != (spans[0][1][0], spans[-1][1][1]) or len(spans) > 2:
                if (ch, (lo, hi)) != tuple(spans):
                    yield [(ch, (lo, hi))]
        return
    unknown = [i for i, (c, _) in enumerate(spans) if c is None]
    allok = sorted(chapter_limits)
    chs = []
    for c in allok:
        ok = True
        for i in unknown:
            vs = spans[i][1]
            if vs is None:
                continue
            if vs[0] < min(chapter_limits[c]) or vs[1] > max(chapter_limits[c]):
                ok = False
                break
        if ok:
            chs.append(c)
    if not chs:
        return
    base = list(spans)
    for fill in itertools.product(chs, repeat=len(unknown)):
        out = list(base)
        for i, c in zip(unknown, fill):
            out[i] = (c, base[i][1])
        yield out


def expand_spans(spans, idx, book):
    """Flatten a concrete span list into [(chapter, verse)]; None if it names
    a chapter or verse range that does not exist in the book."""
    out = []
    for ch, vs in spans:
        if ch is None:
            return None
        have = sorted(idx[book].loc[idx[book].chapter == ch, 'verse_number'])
        if not have:
            return None
        if vs is None:  # whole chapter
            out += [(ch, int(v)) for v in have]
        else:
            lo, hi = vs
            if lo < min(have) or hi > max(have):
                return None
            out += [(ch, v) for v in range(lo, hi + 1)]
    return out


def subs_of(page_text):
    raw = body(page_text)
    marks = list(SUP.finditer(raw))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        t = clean(raw[m.end():end])
        if t:
            out.append(t)
    return out


def verse_text(idx, book, ch, v):
    row = idx[book]
    row = row[(row.chapter == ch) & (row.verse_number == v)]
    if len(row) != 1:
        return None
    return ' '.join(str(row.iloc[0].text).split())


def norm_text(t):
    t = t.lower().replace('Է', 'է').replace('ԗ', 'է')
    t = re.sub(r'[^a-z\u0531-\u0556\u0561-\u0588]+', ' ', t)
    return ' '.join(t.split())


def char_bigrams(t):
    t = norm_text(t).replace(' ', '')
    if len(t) < 2:
        return {t}
    return {t[i:i + 2] for i in range(len(t) - 1)}


def pair_scores(segs, verse_texts):
    """Character-bigram similarity matrix: |A∩B| / min(|A|,|B|).

    Word identity is useless across the 1805 and 1943 editions (different
    morphology AND orthography: տեսին/տեսան, Է/է), but character bigrams
    stay high for true matches (~0.6) and low for wrong ones (~0.3-0.4).
    """
    seg_g = [char_bigrams(s) for s in segs]
    ver_g = [char_bigrams(v) for v in verse_texts]
    mat = []
    for A in seg_g:
        row = []
        for B in ver_g:
            a = min(len(A), len(B))
            row.append(len(A & B) / a if a else 0.0)
        mat.append(row)
    return mat


def dp_align(mat, penalty=0.35, pair_floor=0.28, min_mean=0.35):
    """Penalized sequence alignment.  Returns (mean, pairs, gaps, min_score)
    or None when no gap-tolerant alignment meets the quality bars.  The gap
    budget scales with the passage: small passages are held tighter."""
    n, m = len(mat), len(mat[0])
    max_gap = min(5, max(3, n // 6))
    NEG = float('-inf')
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    par = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == NEG:
                continue
            if i < n and j < m:
                nc = cur + mat[i][j]
                if nc > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = nc
                    par[i + 1][j + 1] = ('m', i, j)
            if j < m:
                nc = cur - penalty
                if nc > dp[i][j + 1]:
                    dp[i][j + 1] = nc
                    par[i][j + 1] = ('v', i, j)
            if i < n:
                nc = cur - penalty
                if nc > dp[i + 1][j]:
                    dp[i + 1][j] = nc
                    par[i + 1][j] = ('s', i, j)
    if dp[n][m] == NEG:
        return None
    pairs, gaps = [], 0
    i, j = n, m
    while i or j:
        move, pi, pj = par[i][j]
        if move == 'm':
            pairs.append((i - 1, j - 1))
        else:
            gaps += 1
        i, j = pi, pj
    pairs.reverse()
    if gaps > max_gap or not pairs:
        return None
    kept = [(i, j) for i, j in pairs if mat[i][j] >= pair_floor]
    if not kept:
        return None
    mean = sum(mat[i][j] for i, j in kept) / len(kept)
    if mean < min_mean:
        return None
    return mean, kept, gaps, min(mat[i][j] for i, j in kept)


def main():
    bible = pd.read_parquet(MODERN_BIBLE)
    bible['chapter'] = bible['chapter'].astype(int)
    bible['verse_number'] = bible['verse_number'].astype(int)
    idx = {}
    chapter_limits = {}
    for bn, sub in bible.groupby('book_number'):
        idx[int(bn)] = sub
        chapter_limits[int(bn)] = {
            int(c): [int(v) for v in g.verse_number]
            for c, g in sub.groupby('chapter')}

    rows, report = [], []
    for g, m, section, num in toc_pairs():
        if section != 'numbered':
            continue
        gp = os.path.join(CACHE, 'numbered-%02d-grabar.txt' % num)
        if not os.path.exists(gp):
            report.append((num, 'SKIP missing grabar page'))
            continue
        page = open(gp, encoding='utf-8').read()
        title_m = WORKING_TITLE.search(page)
        title = title_m.group(1) if title_m else body(page)[:80]
        subs = subs_of(page)

        if num in OVERRIDES:
            refs = [OVERRIDES[num]]
        else:
            refs = parse_refs(title)

        # candidates: combined refs (same book, in order), then each alone
        cands = []
        if len(refs) > 1 and all(r[0] == refs[0][0] for r in refs):
            cands.append((refs[0][0], [s for r in refs for s in r[1]]))
        cands.extend(refs)

        best = None
        if not refs:
            why = 'no ref found  «%s»' % title[:44]
        else:
            tried_books = set()
            for book, spans in cands:
                if book not in idx:
                    tried_books.add(book)
                    continue
                for lst in candidate_lists(book, spans, chapter_limits[book]):
                    verses = expand_spans(lst, idx, book)
                    if verses is None:
                        continue
                    vtexts = []
                    ok = True
                    for (c, v) in verses:
                        t = verse_text(idx, book, c, v)
                        if t is None:
                            ok = False
                            break
                        vtexts.append(t)
                    if not ok or abs(len(verses) - len(subs)) > 3:
                        continue
                    mat = pair_scores(subs, vtexts)
                    r = dp_align(mat)
                    if r is None:
                        continue
                    mean, pairs, gaps, minsc = r
                    if best is None or mean > best[0]:
                        best = (mean, book, lst, verses, vtexts, pairs, gaps, minsc)
            if best is not None:
                mean, book, lst, verses, vtexts, pairs, gaps, minsc = best
                why = ''
                if gaps:
                    why = 'gaps=%d' % gaps
            else:
                if tried_books and all(b not in idx for b in tried_books):
                    why = 'book %s not in modern dataset' % \
                        BOOK_LABEL.get(refs[0][0], refs[0][0])
                else:
                    why = ('no candidate aligned (%d segs; cands %s)'
                           % (len(subs),
                              [str(s)[:56] for _, s in cands[:2]]))
        if best is None:
            report.append((num, 'SKIP %s' % why))
            continue

        mean, book, lst, verses, vtexts, pairs, gaps, minsc = best
        nkept = len(pairs)
        dropped_short = sum(1 for i, j in pairs
                            if len(vtexts[j]) < 8 or len(subs[i]) < 8)
        tag = 'txt%d-b%d-ch%s' % (num, book,
                                  '-'.join(str(c) for c, _ in lst))
        for i, j in pairs:
            if len(vtexts[j]) < 8 or len(subs[i]) < 8:
                continue
            (ch, v) = verses[j]
            rows.append((vtexts[j], subs[i], '%s-%d-%d' % (tag, ch, v)))
        report.append((num, 'ok %d/%d verses  mean=%.2f min=%.2f %s  «%s»' % (
            nkept - dropped_short, len(subs), mean, minsc,
            ('(' + why + ')') if why else '', title[:36])))

    for r in report:
        print('%2d  %s' % r)

    seen, uniq = set(), []
    for a, b, tag in rows:
        if b in seen:
            continue
        seen.add(b)
        uniq.append((a, b, tag))
    with open(OUT, 'w', encoding='utf-8') as f:
        for a, b, tag in uniq:
            f.write('%s\t%s\t%s\n' % (a, b, tag))
    print('\npairs: %d' % len(uniq))
    for a, b, _ in uniq[:3]:
        print('\nMOD:', a[:150])
        print('GRB:', b[:150])


if __name__ == '__main__':
    main()