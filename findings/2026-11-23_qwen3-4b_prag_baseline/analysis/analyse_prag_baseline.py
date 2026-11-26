"""
Tables for the PRAG baseline (experiment 16): one LoRA per wiki2026 document, compared with the experiment-15 runs.

Usage (from the function_vectors root):
python findings/2026-11-23_qwen3-4b_prag_baseline/analysis/analyse_prag_baseline.py findings/2026-11-23_qwen3-4b_prag_baseline/outputs
"""
import json, os, random, re, sys, unicodedata

import pandas as pd

out_dir = sys.argv[1]
EXP15 = 'findings/2026-11-20_qwen3-4b_wiki2026_coverage/outputs'
DATA = 'dataset_files/knowledge/wiki2026'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 100)


def normalise(s):  # as in src/filter_wiki2026_questions.py
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def contains(text, answers):
    return any(f" {normalise(a)} " in f" {normalise(text)} " for a in answers if normalise(a))


def interval(rows):
    """95% bootstrap interval of the accuracy, resampling facts (phrasings of one fact aren't independent)."""
    groups = [g.modified_right.tolist() for _, g in rows.groupby('item_id')]
    rng, means = random.Random(0), []
    for _ in range(2000):
        s = [x for _ in groups for x in rng.choice(groups)]
        means.append(sum(s) / len(s))
    means.sort()
    return f"{means[50]:.2f}-{means[1949]:.2f}"


runs = {}
for r in ['qa_layers8_all', 'qa_doc_layers8_all']:
    runs[f'exp15 {r}'] = pd.read_csv(f'{EXP15}/{r}/results.csv')
for r in ['lora_prag', 'lora_qdoc_c1']:
    df = pd.read_csv(f'{out_dir}/{r}/results.csv')
    for fmt in ['qa', 'chat']:
        runs[f'{r} {fmt}'] = df[df.format == fmt]

rows = []
for name, df in runs.items():
    own = df[df.split == 'own']
    test, held, hop = [own[own.kind == k] for k in ('test', 'held_out_fact', 'two_hop')]
    other, gen = df[df.split == 'held_out_doc'], df[df.split == 'general']
    rows.append({'run': name, 'train': df[df.split == 'train'].modified_right.mean(),
                 'test': f"{test.modified_right.sum()}/{len(test)}", 'test_rate': test.modified_right.mean(), 'test_95%': interval(test),
                 'held_out': f"{held.modified_right.sum()}/{len(held)}", 'held_rate': held.modified_right.mean(), 'held_95%': interval(held),
                 'held_facts_any': f"{held.groupby('item_id').modified_right.any().sum()}/{held.item_id.nunique()}",
                 'two_hop': f"{hop.modified_right.sum()}/{len(hop)}", 'leak': other.gives_own_answer.mean(),
                 'other_unchanged': other.unchanged.mean(), 'general': gen.modified_right.mean(),
                 'general_unchanged': gen.unchanged.mean(), 'general_first_line_words': gen.modified.fillna('').str.split().str.len().mean(),
                 'unmodified_words': gen.unmodified.fillna('').str.split().str.len().mean()})
print("Main table")
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nGenerated questions: do they cover the subagent-written facts? (heuristic: some pair's answer contains the fact's")
print("answer, or the fact's answer contains the pair's answer and the pair's answer is at least 4 characters long)")
aug = {a['doc_id']: a for a in json.load(open(f'{DATA}/prag_augmented.json'))}
filt = json.load(open(f'{DATA}/filtered_v2.json'))
pairs = {d: [x for p in a['passages'] for x in p['qa']] for d, a in aug.items()}
print(f"{sum(len(a['passages']) for a in aug.values())} passages, {sum(map(len, pairs.values()))} pairs")
fact_rows = []
for q in filt:
    for f in q['facts']:
        ans = [f['answer']] + f.get('aliases', [])
        cov = any(contains(x['answer'], ans) or (len(normalise(x['answer'])) >= 4 and contains(' '.join(ans), [x['answer']]))
                  for x in pairs[q['doc_id']])
        fact_rows.append({'item_id': f['fact_id'], 'split': f['split'], 'covered': cov})
facts = pd.DataFrame(fact_rows)
print(facts.groupby('split').covered.agg(['sum', 'count']).to_string())
cover = facts.set_index('item_id').covered
for name in ['lora_prag qa', 'lora_prag chat']:
    own = runs[name][runs[name].split == 'own']
    for kind in ['test', 'held_out_fact']:
        s = own[own.kind == kind]
        s = s.assign(covered=s.item_id.map(cover))
        print(name, kind, s.groupby('covered').modified_right.agg(['sum', 'count']).to_dict('index'))
title_words = {q['doc_id']: {w for w in normalise(aug[q['doc_id']]['title']).split() if len(w) > 3} for q in filt}
n = sum(bool(title_words[d] & set(normalise(x['question']).split())) for d, ps in pairs.items() for x in ps)
print(f"Generated questions sharing a word (>3 letters) with the article title: {n}/{sum(map(len, pairs.values()))}")

print("\nHeld-out phrasings answered right, lora_prag qa")
own = runs['lora_prag qa'][runs['lora_prag qa'].split == 'own']
s = own[(own.kind == 'held_out_fact') & own.modified_right]
print(s[['item_id', 'answers', 'modified']].assign(modified=s.modified.str[:110]).to_string(index=False))

print("\nHeld-out phrasings answered right, lora_qdoc_c1 qa")
own = runs['lora_qdoc_c1 qa'][runs['lora_qdoc_c1 qa'].split == 'own']
s = own[(own.kind == 'held_out_fact') & own.modified_right]
print(s[['item_id', 'answers', 'modified']].assign(modified=s.modified.str[:110]).to_string(index=False))
print("\nTwo-step, lora_qdoc_c1 qa")
s = own[own.kind == 'two_hop']
print(s[['item_id', 'answers', 'modified', 'modified_right']].assign(modified=s.modified.str[:110]).to_string(index=False))

print("\nGeneral answers, lora_prag qa and lora_qdoc_c1 qa (first 4)")
for name in ['lora_prag qa', 'lora_qdoc_c1 qa']:
    s = runs[name][runs[name].split == 'general'].head(4)
    print(s[['question', 'unmodified', 'modified']].assign(modified=s.modified.str[:100], unmodified=s.unmodified.str[:60]).to_string(index=False))

print("\nTest accuracy per document")
print(pd.concat([df[(df.split == 'own') & (df.kind == 'test')].groupby('doc_id').modified_right.mean().rename(n)
                 for n, df in runs.items()], axis=1).round(2).to_string())

print("\nBleed into general answers: 240 general answers that mention the document's title words (>3 letters, not in the question)")
print("(title words taken from the document whose intervention is active; digits such as 2026 excluded)")
titles = {json.loads(l)['doc_id']: json.loads(l)['title'] for l in open(f'{DATA}/documents.jsonl')}
for name, df in runs.items():
    g = df[df.split == 'general']
    def bleeds(r, col):
        words = {w for w in normalise(titles[r.doc_id]).split() if len(w) > 3 and not w.isdigit()} - set(normalise(r.question).split())
        return bool(words & set(normalise(str(r[col])).split()))
    print(f"{name}: modified {g.apply(lambda r: bleeds(r, 'modified'), axis=1).sum()}, unmodified {g.apply(lambda r: bleeds(r, 'unmodified'), axis=1).sum()} of {len(g)}")

print("\nScoring check: right answers that stay right if only the first 12 words count (lenient first-line scoring vs stricter)")
for name, df in runs.items():
    own = df[df.split == 'own']
    out = []
    for kind in ['test', 'held_out_fact']:
        s = own[own.kind == kind]
        strict = s.apply(lambda r: contains(' '.join(str(r.modified).split()[:12]), r.answers.split('|')), axis=1)
        out.append(f"{kind} {s.modified_right.sum()} -> {strict.sum()} of {len(s)}")
    print(name, '; '.join(out))
