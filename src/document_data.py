"""
Shared data and evaluation helpers for per-document knowledge storage on the wiki2026 dataset, used by
learn_document_intervention.py (learned vectors / key-value memory) and learn_document_lora.py (PRAG-style LoRA).

Training items come from one of two sources:
- 'questions': the subagent-written training questions of the document's trained facts (filtered.json / filtered_v2.json);
- 'prag': the question-answer pairs Qwen3-4B generated from the document's passages (augment_documents_prag.py).
Test items are always the subagent-written ones: test phrasings of trained facts, all phrasings of held-out facts and
two-step questions.
"""
import json

import pandas as pd

from learn_knowledge_vector import KNOWN_FACTS
from eval_contrast_interventions import GENERAL_CITY, GENERAL_OTHER
from filter_wiki2026_questions import is_right


def load_documents(data_dir, questions_file):
    """Documents by id, and the question data (documents with at least one kept fact), sorted by document id."""
    docs = {d['doc_id']: d for d in map(json.loads, open(f'{data_dir}/documents.jsonl'))}
    data = sorted((q for q in json.load(open(f'{data_dir}/{questions_file}')) if q['facts']), key=lambda q: q['doc_id'])
    return docs, data


def answers_of(x):
    return [x['answer']] + x.get('aliases', [])


def trained_facts(q):
    return [f for f in q['facts'] if f.get('split', 'trained') == 'trained']


def held_out_facts(q):
    return [f for f in q['facts'] if f.get('split', 'trained') == 'held_out']


def make_train_items(source, augmented=None):
    """
    Returns train_items(q) -> [(question, answer, [answers], item_id)]. For 'prag', every generated question-answer pair
    of the document's passages (augmented: {doc_id: augmentation record}).
    """
    if source == 'questions':
        return lambda q: [(t, f['answer'], answers_of(f), f"{f['fact_id']}_train{j}")
                          for f in trained_facts(q) for j, t in enumerate(f['train'])]
    assert source == 'prag' and augmented is not None
    return lambda q: [(a['question'], a['answer'], [a['answer']], f"{q['doc_id']}_{p['pid']}_qa{j}")
                      for p in augmented[q['doc_id']]['passages'] for j, a in enumerate(p['qa'])]


def test_items(q):
    """[(question, [answers], kind, item_id)]: test phrasings, held-out facts (all phrasings), two-step questions."""
    items = [(f['test'], answers_of(f), 'test', f['fact_id']) for f in trained_facts(q)]
    items += [(t, answers_of(f), 'held_out_fact', f['fact_id']) for f in held_out_facts(q) for t in f['train'] + [f['test']]]
    return items + [(h['question'], answers_of(h), 'two_hop', h['q_id']) for h in q['two_hop']]


def general_test_items():
    pairs = [(q, [a]) for q, a in KNOWN_FACTS] + [(q, list(a)) for q, a in GENERAL_CITY + GENERAL_OTHER]
    return [(q, answers, 'general', f'g{i}') for i, (q, answers) in enumerate(pairs)]


def evaluate_document(q, held_out_docs, train_items, prompt, answer_modified, answer_unmodified, unmodified_cache,
                      extra=None):
    """
    Scores one document's storage: its training items, its own test items, the held-out documents' test items and the
    general questions. answer_*(prompts) -> first-line answers; unmodified answers are cached by prompt.
    Returns one row per question.
    """
    eval_sets = [('train', [(t, answers, 'train', item_id) for t, _, answers, item_id in train_items(q)], q),
                 ('own', test_items(q), q)]
    eval_sets += [('held_out_doc', test_items(h), h) for h in held_out_docs]
    eval_sets += [('general', general_test_items(), None)]
    own_answers = [f['answer'] for f in q['facts']]
    rows = []
    for split, items, asked_doc in eval_sets:
        prompts = [prompt(t) for t, _, _, _ in items]
        modified = answer_modified(prompts)
        missing = list(dict.fromkeys(p for p in prompts if p not in unmodified_cache))
        if missing:
            unmodified_cache.update(zip(missing, answer_unmodified(missing)))
        for (t, answers, kind, item_id), p, m in zip(items, prompts, modified):
            u = unmodified_cache[p]
            rows.append({'doc_id': q['doc_id'], 'split': split, 'asked_doc': asked_doc['doc_id'] if asked_doc else None,
                         'kind': kind, 'item_id': item_id, 'question': t, 'answers': '|'.join(answers),
                         'modified': m, 'unmodified': u, 'modified_right': is_right(m, answers),
                         'unmodified_right': is_right(u, answers), 'unchanged': m == u,
                         'gives_own_answer': is_right(m, own_answers) and split not in ('own', 'train'), **(extra or {})})
    return rows


def summarise(df):
    """The headline metrics of one run (one evaluation format)."""
    g = lambda split, kind=None: df[(df.split == split) & ((df.kind == kind) if kind else True)]
    mean = lambda s: float(s.mean()) if len(s) else None
    return {'train_right': mean(g('train').modified_right),
            'test_right': mean(g('own', 'test').modified_right),
            'two_hop_right': mean(g('own', 'two_hop').modified_right),
            'held_out_fact_right': mean(g('own', 'held_out_fact').modified_right),
            'held_out_fact_right_unmodified': mean(g('own', 'held_out_fact').unmodified_right),
            'test_right_unmodified': mean(g('own', 'test').unmodified_right),
            'two_hop_right_unmodified': mean(g('own', 'two_hop').unmodified_right),
            'held_out_docs_unchanged': mean(g('held_out_doc').unchanged),
            'held_out_docs_gives_own_answer': mean(g('held_out_doc').gives_own_answer),
            'held_out_docs_right': mean(g('held_out_doc').modified_right),
            'general_right': mean(g('general').modified_right),
            'general_right_unmodified': mean(g('general').unmodified_right),
            'general_unchanged': mean(g('general').unchanged)}
