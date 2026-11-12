"""Bulk-fetch the «Գրաբարի դասագիրք» pages through the MediaWiki API.

The per-page ``action=raw`` endpoint throttles hard (≈1 page/minute), while a
single ``prop=revisions`` query returns up to 50 pages.  Same cache layout as
fetch_dasagirq.py, so either fetcher can fill the cache.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_dasagirq import CACHE, BASE, pairs, cache_path

UA = 'ArmenianMTResearch/1.0 (open-source MT dataset building)'
API = 'https://hy.wikisource.org/w/api.php'


def bulk(titles):
    params = {'action': 'query', 'prop': 'revisions', 'rvprop': 'content',
              'rvslots': 'main', 'format': 'json', 'formatversion': '2',
              'titles': '|'.join(titles), 'redirects': '1'}
    url = API + '?' + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception as e:
            print('  retry:', e)
            time.sleep(10 + 10 * attempt)
    return {}


def main():
    wanted = []
    for g, m, section, num in pairs():
        for side, title in (('grabar', g), ('modern', m)):
            path = cache_path(section, num, side)
            if not os.path.exists(path):
                wanted.append((BASE + title.replace('_', ' '), path))
    print('missing pages:', len(wanted))
    os.makedirs(CACHE, exist_ok=True)

    done = 0
    for i in range(0, len(wanted), 20):
        chunk = wanted[i:i + 20]
        titles = [t for t, _ in chunk]
        print('batch %d-%d' % (i, i + len(chunk)))
        data = bulk(titles)
        pages = {p.get('title'): p for p in data.get('query', {}).get('pages', [])}
        # redirects: normalized/redirected titles resolve to the target
        for p in data.get('query', {}).get('pages', []):
            pass
        norm = {}
        for n in data.get('query', {}).get('normalized', []) + data.get('query', {}).get('redirects', []):
            norm[n['from']] = n['to']
        for title, path in chunk:
            t = title
            for _ in range(3):
                t = norm.get(t, t)
            page = pages.get(t) or pages.get(title)
            ok = False
            if page and 'revisions' in page:
                content = page['revisions'][0]['slots']['main']['content']
                if len(content) > 300:
                    open(path, 'w', encoding='utf-8').write(content)
                    done += 1
                    ok = True
            print('  %-55s %s' % (title[-55:], 'ok' if ok else 'MISSING'))
        time.sleep(2)
    print('fetched', done, 'of', len(wanted))


if __name__ == '__main__':
    main()
