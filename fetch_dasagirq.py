"""Fetch the bilingual pages of Manuk Abeghyan's «Գրաբարի դասագիրք» (1916).

The textbook's table of contents lists the same 37 numbered passages twice —
once in Grabar, once in Modern Armenian — plus eight chapters of Ełišē's
«Վասն Վարդանայ եւ Հայոց պատերազմին» in both versions.  Those are real human
translations of the same text, so they extend the Bible corpus with a second
register (homiletics, psalms, historiography).

Pages are cached under data/wikisource/dasagirq/ so the build is re-runnable
offline.  Requests are throttled: hy.wikisource answers 429 when hammered.

    python3 fetch_dasagirq.py            # download missing pages
    python3 fetch_dasagirq.py --list     # just show the pairing
"""
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data', 'wikisource', 'dasagirq')
INDEX = os.path.join(HERE, 'data', 'wikisource', 'dasagirq_index.txt')
UA = 'ArmenianMTResearch/1.0 (open-source MT dataset building)'
BASE = 'Գրաբարի դասագիրք/'

LINK = re.compile(r'\[\[/([^|\]]+)\|([^\]]+)\]\]')
NUMBERED = re.compile(r'(\d+)\s*[.․]')


def _label(line):
    m = LINK.search(line)
    return m.group(2).strip() if m else line.lstrip('*').strip()


def index_entries(path=INDEX):
    """Return [(section, number, title)] in table-of-contents order."""
    text = open(path, encoding='utf-8').read()
    sections = []
    cur = None
    for line in text.splitlines():
        if line.startswith('*') and not line.startswith('**'):
            cur = _label(line)
            sections.append((cur, []))
        elif line.startswith('**') and sections:
            m = LINK.search(line)
            if m:
                sections[-1][1].append((m.group(1), _label(line)))

    out = []
    for name, items in sections:
        for i, (title, label) in enumerate(items, 1):
            m = NUMBERED.match(label)
            num = int(m.group(1)) if m else i
            out.append((name, num, title))
    return out


def pairs():
    """[(grabar_title, modern_title, section, number)] for parallel passages."""
    entries = index_entries()
    grabar_num, modern_num, grabar_el, modern_el = {}, {}, [], []
    for section, num, title in entries:
        if 'աշխարհաբար' in section:
            modern_num[num] = title
        elif 'գրաբար' in section:
            grabar_num[num] = title
        elif 'վասն Վարդանանց' in section:
            grabar_el.append(title)
        elif 'Վարդանանցի' in section:
            modern_el.append(title)

    out = []
    for num in sorted(set(grabar_num) & set(modern_num)):
        section = 'numbered'
        out.append((grabar_num[num], modern_num[num], section, num))
    for i, (g, m) in enumerate(zip(grabar_el, modern_el), 1):
        out.append((g, m, 'elishe', i))
    return out


def cache_path(section, num, side):
    return os.path.join(CACHE, '%s-%02d-%s.txt' % (section, num, side))


def fetch(title, path):
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return open(path, encoding='utf-8').read()
    url = 'https://hy.wikisource.org/w/index.php?' + urllib.parse.urlencode(
        {'title': BASE + title, 'action': 'raw'})
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            txt = urllib.request.urlopen(req, timeout=90).read().decode('utf-8')
            if len(txt) > 300:
                os.makedirs(CACHE, exist_ok=True)
                open(path, 'w', encoding='utf-8').write(txt)
                return txt
            print('  short page:', title, len(txt))
        except Exception as e:
            print('  err:', title, e)
        time.sleep(8 + 6 * attempt)
    return ''


def main():
    ps = pairs()
    print('parallel passages:', len(ps))
    if '--list' in sys.argv:
        for g, m, s, n in ps:
            print('%-8s %2d  %-45s | %s' % (s, n, g[:45], m[:45]))
        return

    missing_before = 0
    for g, m, section, num in ps:
        for side, title in (('grabar', g), ('modern', m)):
            path = cache_path(section, num, side)
            if not os.path.exists(path):
                missing_before += 1
                txt = fetch(title, path)
                print('%-8s %02d %-7s %5d chars  %s'
                      % (section, num, side, len(txt), title[:45]))
                time.sleep(5)
    print('fetched', missing_before, 'pages; cache:', CACHE)


if __name__ == '__main__':
    main()
