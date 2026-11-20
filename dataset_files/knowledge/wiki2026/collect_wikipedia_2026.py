"""
Collects English Wikipedia articles about 2026 events, as documents the model cannot have seen in training
(Qwen3's technical report is from May 2025).

An article is kept only if both hold:
- it is in a 2026 event category (the roots below or their subcategories, up to depth 2), and
- its first revision is from 2026 or later, so the article itself (not only an edit) is new.

Text is Wikipedia's plain-text extract (CC BY-SA 4.0); title, URL, page id and revision id are kept for attribution.

Usage (from the repo root): python dataset_files/knowledge/wiki2026/collect_wikipedia_2026.py
Writes dataset_files/knowledge/wiki2026/candidates.jsonl. API responses are cached in results/wiki2026_cache,
so an interrupted run resumes without repeating requests.
"""
import argparse, hashlib, json, os, random, re, time, urllib.error, urllib.parse, urllib.request

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "fv-knowledge-dataset/0.1 (university research project; contact: wikipedia.lesser377@passinbox.com)"
ROOTS = ["Category:2026 disasters", "Category:2026 elections", "Category:2026 in science", "Category:2026 in sports",
         "Category:2026 in politics", "Category:2026 in technology", "Category:2026 in international relations",
         "Category:2026 in economics", "Category:2026 in business"]
DROP_SECTIONS = {"See also", "References", "External links", "Notes", "Further reading", "Citations", "Sources",
                 "Bibliography", "Footnotes"}
SKIP_TITLE = re.compile(r"^(List of|Lists of|Timeline of|Results of|Opinion polling|Polling for)", re.I)


class Client:
    def __init__(self, cache_dir, delay):
        self.cache_dir, self.delay, self.last = cache_dir, delay, 0.0
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, params):
        params = {**params, 'format': 'json', 'formatversion': 2, 'maxlag': 5}
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = f"{self.cache_dir}/{key}.json"
        if os.path.exists(path):
            return json.load(open(path))
        url = API + '?' + urllib.parse.urlencode(params)
        for attempt in range(8):
            time.sleep(max(0.0, self.delay - (time.time() - self.last)))
            self.last = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=60) as r:
                    data = json.load(r)
                if data.get('error', {}).get('code') == 'maxlag':
                    time.sleep(10)
                    continue
                json.dump(data, open(path, 'w'))
                return data
            except urllib.error.HTTPError as e:
                wait = int(e.headers.get('Retry-After', 30)) if e.code == 429 else 10 * (attempt + 1)
                print(f"HTTP {e.code}, waiting {wait} s", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"{e}, retrying", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"Request failed: {params}")


def crawl(client, roots, max_depth):
    pages, seen, frontier = {}, set(), [(r, 0) for r in roots]
    while frontier:
        cat, depth = frontier.pop()
        if cat in seen:
            continue
        seen.add(cat)
        cont = {}
        while True:
            d = client.get({'action': 'query', 'list': 'categorymembers', 'cmtitle': cat, 'cmlimit': 500,
                            'cmtype': 'page|subcat', **cont})
            for m in d.get('query', {}).get('categorymembers', []):
                if m['ns'] == 14 and depth < max_depth:
                    frontier.append((m['title'], depth + 1))
                elif m['ns'] == 0:
                    pages.setdefault(m['title'], cat)
            if 'continue' not in d:
                break
            cont = d['continue']
    return pages, seen


def clean_extract(text):
    """Drops reference-type sections from a plain-text extract (section headings look like '== Title ==')."""
    out, keep = [], True
    for line in text.split('\n'):
        heading = re.match(r"^(=+)\s*(.*?)\s*\1$", line.strip())
        if heading:
            keep = heading.group(2) not in DROP_SECTIONS
        if keep:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", '\n'.join(out)).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='dataset_files/knowledge/wiki2026/candidates.jsonl')
    parser.add_argument('--cache_dir', default='results/wiki2026_cache')
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--min_bytes', type=int, default=8000, help='Minimum wikitext size before fetching details')
    parser.add_argument('--created_after', default='2026-01-01T00:00:00Z')
    parser.add_argument('--delay', type=float, default=1.0, help='Seconds between requests')
    parser.add_argument('--max_kept', type=int, default=300,
                        help='Stop after this many qualifying articles; candidates are checked in a random order (--seed)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    client = Client(args.cache_dir, args.delay)

    pages, categories = crawl(client, ROOTS, args.max_depth)
    titles = sorted(t for t in pages if not SKIP_TITLE.match(t))
    print(f"{len(categories)} categories, {len(pages)} pages, {len(titles)} after title filter", flush=True)

    info = {}
    for i in range(0, len(titles), 50):
        d = client.get({'action': 'query', 'prop': 'info', 'titles': '|'.join(titles[i:i + 50])})
        for p in d['query']['pages']:
            if not p.get('missing') and p.get('length', 0) >= args.min_bytes:
                info[p['title']] = p
    print(f"{len(info)} pages with at least {args.min_bytes} bytes", flush=True)

    order = sorted(info.items())
    random.Random(args.seed).shuffle(order)  # so stopping at --max_kept doesn't bias the selection alphabetically
    kept = []
    for n, (title, p) in enumerate(order):
        if len(kept) >= args.max_kept:
            break
        d = client.get({'action': 'query', 'prop': 'revisions', 'pageids': p['pageid'], 'rvdir': 'newer', 'rvlimit': 1,
                        'rvprop': 'timestamp|ids'})
        created = d['query']['pages'][0]['revisions'][0]['timestamp']
        if created < args.created_after:
            continue
        d = client.get({'action': 'query', 'prop': 'extracts|revisions', 'pageids': p['pageid'], 'explaintext': 1,
                        'exsectionformat': 'wiki', 'rvprop': 'ids|timestamp'})
        page = d['query']['pages'][0]
        text = clean_extract(page.get('extract', ''))
        kept.append({'title': title, 'url': 'https://en.wikipedia.org/wiki/' + urllib.parse.quote(title.replace(' ', '_')),
                     'pageid': p['pageid'], 'revid': page['revisions'][0]['revid'],
                     'revision_timestamp': page['revisions'][0]['timestamp'], 'created': created,
                     'category': pages[title], 'n_words': len(text.split()), 'text': text,
                     'license': 'CC BY-SA 4.0'})
        if (n + 1) % 100 == 0:
            print(f"checked {n + 1}/{len(info)}, kept {len(kept)}", flush=True)

    with open(args.out, 'w') as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Checked {n + 1} of {len(info)} candidates; kept {len(kept)} articles created on or after "
          f"{args.created_after}; wrote {args.out}")
