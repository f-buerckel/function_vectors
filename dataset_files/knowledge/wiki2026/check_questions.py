"""
Checks the question files in questions/ against the rules in question_instructions.md (format and text checks only;
whether the model can answer is checked separately on the GPU).

Usage (from the repo root): python dataset_files/knowledge/wiki2026/check_questions.py [doc_id ...]
"""
import json, os, re, sys

HERE = 'dataset_files/knowledge/wiki2026'
VAGUE = re.compile(r"\b(the|this) (event|article|document|text|tournament|competition|incident|war|attack|election|game)\b(?! of)", re.I)


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {'the', 'a', 'an', 'of', 'in', 'at', 'on', 'to', 'was', 'is', 'what',
                                                         'which', 'who', 'did', 'does', 'for', 'by', 'and', 'how', 'many'}


def overlap(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(1, len(a | b))


def check(doc, q):
    problems = []
    text, low = doc['text'], doc['text'].lower()
    if len(q.get('facts', [])) != 5 or len(q.get('two_hop', [])) != 2:
        problems.append(f"expected 5 facts and 2 two-hop questions, got {len(q.get('facts', []))} and {len(q.get('two_hop', []))}")
    for f in q.get('facts', []):
        fid, answers = f['fact_id'], [f['answer']] + f.get('aliases', [])
        if f['evidence'] not in text:
            problems.append(f"{fid}: evidence not found verbatim in document")
        if f['answer'].lower() not in f['evidence'].lower():
            problems.append(f"{fid}: answer not in evidence")
        if len(f['train']) != 2:
            problems.append(f"{fid}: expected 2 training questions")
        for question in f['train'] + [f['test']]:
            if not question.strip().endswith('?'):
                problems.append(f"{fid}: question doesn't end with '?': {question}")
            if any(a.lower() in question.lower() for a in answers if len(a) > 2):
                problems.append(f"{fid}: question contains the answer: {question}")
            if VAGUE.search(question):
                problems.append(f"{fid}: vague reference ('{VAGUE.search(question).group(0)}'): {question}")
        o = max(overlap(f['test'], t) for t in f['train'])
        if o > 0.6:
            problems.append(f"{fid}: test wording too close to a training question (word overlap {o:.2f})")
    for h in q.get('two_hop', []):
        hid = h['q_id']
        if h['evidence'] not in text:
            problems.append(f"{hid}: evidence not found verbatim in document")
        for a in [h['answer']] + h.get('aliases', []):
            if len(a) > 2 and re.search(r"\b" + re.escape(a.lower()) + r"\b", low):
                problems.append(f"{hid}: two-hop answer '{a}' appears in the document")
        if any(a.lower() in h['question'].lower() for a in [h['answer']] + h.get('aliases', []) if len(a) > 2):
            problems.append(f"{hid}: question contains the answer")
        if VAGUE.search(h['question']):
            problems.append(f"{hid}: vague reference: {h['question']}")
    return problems


if __name__ == "__main__":
    docs = {d['doc_id']: d for d in map(json.loads, open(f'{HERE}/documents.jsonl'))}
    ids = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(f'{HERE}/questions') if f.endswith('.json'))
    total, n_questions = 0, 0
    for doc_id in ids:
        q = json.load(open(f'{HERE}/questions/{doc_id}.json'))
        n_questions += sum(len(f['train']) + 1 for f in q['facts']) + len(q['two_hop'])
        problems = check(docs[doc_id], q)
        total += len(problems)
        print(f"{doc_id} {docs[doc_id]['title']}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        for p in problems:
            print(f"    {p}")
    print(f"\n{len(ids)} files, {n_questions} questions, {total} problems")
