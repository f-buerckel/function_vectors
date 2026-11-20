"""
Filters the wiki2026 questions with the model itself:
- closed-book: the question alone (Q:/A: format). A question the model already answers right doesn't test the document.
- in-context: the document, then the question. A question the model can't answer even with the document is badly
  posed or too hard, and is dropped.

An answer counts as right if the greedy answer's first line contains the reference answer or an alias (case-insensitive,
ignoring punctuation and accents).

Keep rules:
- fact: no phrasing (2 training, 1 test) answered closed-book; the test phrasing and at least one training phrasing
  answered in context. Training phrasings not answered in context are dropped from the fact.
- two-step question: not answered closed-book, answered in context.

Usage (from src/): python filter_wiki2026_questions.py
Writes ../dataset_files/knowledge/wiki2026/filter_results.csv and filtered.json.
"""
import argparse, json, os, re, unicodedata

import pandas as pd
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed


def normalise(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def is_right(generated, answers):
    g = f" {normalise(generated)} "
    return any(f" {normalise(a)} " in g for a in answers if normalise(a))


def generate(model, tokenizer, prompts, max_new_tokens, batch_size):
    tokenizer.padding_side = 'left'
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        out += [t.strip().split('\n')[0].strip() for t in texts]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_dir', type=str, default='../dataset_files/knowledge/wiki2026')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16, help='closed-book batch size')
    parser.add_argument('--context_batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    set_seed(42)
    docs = {d['doc_id']: d for d in map(json.loads, open(f'{args.data_dir}/documents.jsonl'))}
    files = sorted(f for f in os.listdir(f'{args.data_dir}/questions') if f.endswith('.json'))
    questions = [json.load(open(f'{args.data_dir}/questions/{f}')) for f in files]

    rows = []
    for q in questions:
        for f in q['facts']:
            answers = [f['answer']] + f.get('aliases', [])
            for kind, text in [('train', f['train'][0]), ('train', f['train'][1]), ('test', f['test'])]:
                rows.append({'doc_id': q['doc_id'], 'item_id': f['fact_id'], 'kind': kind, 'question': text, 'answers': answers})
        for h in q['two_hop']:
            rows.append({'doc_id': q['doc_id'], 'item_id': h['q_id'], 'kind': 'two_hop', 'question': h['question'],
                         'answers': [h['answer']] + h.get('aliases', [])})

    closed = generate(model, tokenizer, [f"Q: {r['question']}\nA:" for r in rows], args.max_new_tokens, args.batch_size)
    context_prompts = [f"{docs[r['doc_id']]['title']}\n\n{docs[r['doc_id']]['text']}\n\nQ: {r['question']}\nA:" for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(context_prompts[i]))  # similar lengths per batch, less padding
    in_context = [None] * len(rows)
    for i, text in zip(order, generate(model, tokenizer, [context_prompts[i] for i in order], args.max_new_tokens,
                                       args.context_batch_size)):
        in_context[i] = text
    for r, c, ic in zip(rows, closed, in_context):
        r.update({'closed_book_answer': c, 'in_context_answer': ic, 'closed_book_right': is_right(c, r['answers']),
                  'in_context_right': is_right(ic, r['answers'])})

    df = pd.DataFrame(rows)
    df.assign(answers=df.answers.map('|'.join)).to_csv(f'{args.data_dir}/filter_results.csv', index=False)

    status = {}
    for item_id, g in df.groupby('item_id'):
        if g.kind.iloc[0] == 'two_hop':
            status[item_id] = {'keep': (not g.closed_book_right.any()) and bool(g.in_context_right.all())}
        else:
            train_ok = g[(g.kind == 'train') & g.in_context_right].question.tolist()
            test_ok = bool(g[g.kind == 'test'].in_context_right.all())
            status[item_id] = {'keep': (not g.closed_book_right.any()) and test_ok and len(train_ok) >= 1,
                               'train_kept': train_ok}
    filtered = []
    for q in questions:
        facts = [{**f, 'train': status[f['fact_id']]['train_kept']} for f in q['facts'] if status[f['fact_id']]['keep']]
        two_hop = [h for h in q['two_hop'] if status[h['q_id']]['keep']]
        filtered.append({'doc_id': q['doc_id'], 'title': q['title'], 'facts': facts, 'two_hop': two_hop})
    json.dump(filtered, open(f'{args.data_dir}/filtered.json', 'w'), indent=1, ensure_ascii=False)

    summary = df.groupby('kind')[['closed_book_right', 'in_context_right']].mean().round(3)
    print(summary.to_string())
    n_facts = sum(len(q['facts']) for q in questions)
    n_hop = sum(len(q['two_hop']) for q in questions)
    kept_facts = sum(len(q['facts']) for q in filtered)
    kept_hop = sum(len(q['two_hop']) for q in filtered)
    print(f"Facts kept: {kept_facts}/{n_facts}; two-step kept: {kept_hop}/{n_hop}; "
          f"questions kept: {sum(len(f['train']) + 1 for q in filtered for f in q['facts']) + kept_hop}/{len(df)}")
