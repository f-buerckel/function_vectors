"""
PRAG-style data augmentation (Su et al., 2025, "Parametric Retrieval Augmented Generation") for the wiki2026 documents,
reimplemented for this project (the official code has no licence and is built around question benchmarks).

Each document is split into passages of at most --max_words words (sentences are not split; a passage doesn't cross a
section heading). For every passage the model itself (Qwen3-4B, chat template, thinking off) writes
- one rewrite that keeps all details, and
- three question-answer pairs, each with a short answer and a full-sentence answer.

Differences from PRAG, on purpose:
- passages come from one long document instead of retrieved 100-word Wikipedia passages;
- the prompts are our own wording. The question prompt adds the article title and asks for self-contained questions
  that name the subject, because our test asks the questions without any passage in context;
- greedy decoding (reproducible); a passage whose questions can't be parsed is retried with sampling (temperature 0.7,
  top-k 50, as in PRAG's generation config), at most --retries times, and keeps no questions if all attempts fail.

Usage (from src/): python augment_documents_prag.py [--questions filtered_v2.json]
Writes ../dataset_files/knowledge/wiki2026/prag_augmented.json: one record per document,
{doc_id, title, passages: [{pid, section, text, rewrite, qa: [{question, answer, full_answer}]}]}.
"""
import argparse, json, re

import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed

REWRITE_PROMPT = """The following passage is from the Wikipedia article "{title}".

Rewrite the passage in different words. Keep every name, place, number, date and other detail exactly as it is. Output only the rewritten passage.

Passage:
{passage}"""

QA_PROMPT = """The following passage is from the Wikipedia article "{title}".

Write three different questions about facts stated in this passage. Each question must be self-contained: it is read without the passage, so every question must name the subject of the article explicitly, for example "In the {title}, …", never "the passage", "the article" or "this event". Each answer must be short (a name, place, number, date or a few words) and copied from the passage.

Output only a JSON list of three objects with the keys "question", "answer" and "full_answer" (the answer as a full sentence).

Passage:
{passage}"""


def split_passages(text, max_words):
    """[(section, passage)]: sentences packed into passages of at most max_words words, within one section."""
    sections, current = [], ('Introduction', [])
    for line in text.split('\n'):
        heading = re.fullmatch(r'\s*=+\s*(.*?)\s*=+\s*', line)
        if heading:
            sections.append(current)
            current = (heading.group(1), [])
        elif line.strip():
            current[1].append(line.strip())
    sections.append(current)
    passages = []
    for section, lines in sections:
        sentences = [s for line in lines for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])', line) if s.strip()]
        chunk = []
        for s in sentences:
            if chunk and len(' '.join(chunk + [s]).split()) > max_words:
                passages.append((section, ' '.join(chunk)))
                chunk = []
            chunk.append(s)
        if chunk:
            passages.append((section, ' '.join(chunk)))
    return passages


def parse_qa(text):
    """Three valid question-answer dicts, or None."""
    try:
        start, end = text.index('['), text.rindex(']') + 1
        items = json.loads(text[start:end])
    except ValueError:
        return None
    keys = ('question', 'answer', 'full_answer')
    items = [{k: str(it[k]).strip() for k in keys} for it in items
             if isinstance(it, dict) and all(str(it.get(k, '')).strip() for k in keys)]
    return items[:3] if len(items) >= 3 else None


def chat(model, tokenizer, messages, max_new_tokens, batch_size, sample=False):
    tokenizer.padding_side = 'left'
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': m}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for m in messages]
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors='pt', padding=True,
                          add_special_tokens=False).to(model.device)
        options = dict(do_sample=True, temperature=0.7, top_k=50) if sample else dict(do_sample=False)
        with torch.no_grad():
            ids = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **options)
        out += [t.strip() for t in tokenizer.batch_decode(ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)]
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_dir', type=str, default='../dataset_files/knowledge/wiki2026')
    parser.add_argument('--questions', type=str, default='filtered_v2.json', help='which documents to augment')
    parser.add_argument('--out', type=str, default='prag_augmented.json')
    parser.add_argument('--max_words', type=int, default=120)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model, tokenizer, _ = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    set_seed(args.seed)
    docs = {d['doc_id']: d for d in map(json.loads, open(f'{args.data_dir}/documents.jsonl'))}
    doc_ids = sorted(q['doc_id'] for q in json.load(open(f'{args.data_dir}/{args.questions}')))[:args.max_docs]

    records = [{'doc_id': d, 'title': docs[d]['title'],
                'passages': [{'pid': f'p{k:02d}', 'section': sec, 'text': text}
                             for k, (sec, text) in enumerate(split_passages(docs[d]['text'], args.max_words))]}
               for d in doc_ids]
    flat = [(r, p) for r in records for p in r['passages']]
    print(f"{len(records)} documents, {len(flat)} passages", flush=True)

    print("Rewrites", flush=True)
    for (r, p), text in zip(flat, chat(model, tokenizer, [REWRITE_PROMPT.format(title=r['title'], passage=p['text'])
                                                          for r, p in flat], args.max_new_tokens, args.batch_size)):
        p['rewrite'] = text
    print("Questions", flush=True)
    todo = flat
    for attempt in range(args.retries + 1):
        texts = chat(model, tokenizer, [QA_PROMPT.format(title=r['title'], passage=p['text']) for r, p in todo],
                     args.max_new_tokens, args.batch_size, sample=attempt > 0)
        for (r, p), text in zip(todo, texts):
            p['qa'], p['qa_attempts'] = parse_qa(text), attempt + 1
        todo = [(r, p) for r, p in todo if p['qa'] is None]
        print(f"attempt {attempt + 1}: {len(todo)} passages without valid questions", flush=True)
        if not todo:
            break
    for r, p in todo:
        p['qa'] = []

    json.dump(records, open(f'{args.data_dir}/{args.out}', 'w'), indent=1, ensure_ascii=False)
    n_qa = sum(len(p['qa']) for _, p in flat)
    print(f"{n_qa} question-answer pairs from {len(flat)} passages; {len(todo)} passages without questions")
