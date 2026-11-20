"""
Selects the documents for the wiki2026 dataset from candidates.jsonl (see collect_wikipedia_2026.py).

Rules (agreed 2026-11-20):
- at least 800 words of plain text;
- not a page that is mostly statistics or lists, and centred on a 2026 event (EXCLUDE);
- at most two articles per overlapping topic (TOPICS; the longest ones are kept);
- text cut at about 3,000 words at a paragraph boundary, so a document fits in the model's context.

Usage (from the repo root): python dataset_files/knowledge/wiki2026/select_documents.py
Writes dataset_files/knowledge/wiki2026/documents.jsonl.
"""
import argparse, json, re

EXCLUDE = {
    "2026 Men's Rugby League World Cup statistics": "mostly statistics tables",
    "2026 NSWRL Major Age Based Competitions": "mostly results lists",
    "2026 in Rizin Fighting Federation": "list of events",
    "Susy Aponte": "biography, mostly life history before 2026",
}
TOPICS = {"Iran war and protests": re.compile(r"\bIran", re.I)}  # matched against title and category
MAX_PER_TOPIC = 2


def truncate(text, max_words):
    """Keeps whole paragraphs up to about max_words words."""
    out, n = [], 0
    for paragraph in text.split('\n'):
        words = len(paragraph.split())
        if n + words > max_words and n > 0:
            break
        out.append(paragraph)
        n += words
    return '\n'.join(out).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default='dataset_files/knowledge/wiki2026/candidates.jsonl')
    parser.add_argument('--out', default='dataset_files/knowledge/wiki2026/documents.jsonl')
    parser.add_argument('--min_words', type=int, default=800)
    parser.add_argument('--max_words', type=int, default=3000)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.candidates)]
    rows = [r for r in rows if r['n_words'] >= args.min_words and r['title'] not in EXCLUDE]
    rows.sort(key=lambda r: -r['n_words'])
    kept, per_topic = [], {t: 0 for t in TOPICS}
    for r in rows:
        topics = [t for t, rx in TOPICS.items() if rx.search(r['title']) or rx.search(r['category'])]
        if any(per_topic[t] >= MAX_PER_TOPIC for t in topics):
            print(f"dropped (topic limit): {r['title']}")
            continue
        for t in topics:
            per_topic[t] += 1
        kept.append(r)

    kept.sort(key=lambda r: r['title'])
    with open(args.out, 'w') as f:
        for i, r in enumerate(kept):
            text = truncate(r['text'], args.max_words)
            doc = {'doc_id': f"d{i:03d}", **{k: r[k] for k in ['title', 'url', 'pageid', 'revid', 'revision_timestamp',
                                                              'created', 'category', 'license']},
                   'n_words_full': r['n_words'], 'n_words': len(text.split()), 'truncated': len(text.split()) < r['n_words'],
                   'text': text}
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    words = sorted(len(truncate(r['text'], args.max_words).split()) for r in kept)
    print(f"{len(kept)} documents; words after truncation: min {words[0]}, median {words[len(words) // 2]}, max {words[-1]}")
