"""
Tables for the wiki2026 pilot: per-document interventions on 10 real documents (147 filtered questions).

Usage (from the function_vectors root):
python findings/2026-11-20_qwen3-4b_wiki2026_pilot/analysis/analyse_wiki2026_pilot.py findings/2026-11-20_qwen3-4b_wiki2026_pilot/outputs
"""
import glob, json, os, sys

import pandas as pd

out_dir = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 90)
filtered = {q['doc_id']: q for q in json.load(open('dataset_files/knowledge/wiki2026/filtered.json'))}

rows, per_doc, frames = [], [], {}
for f in sorted(glob.glob(f'{out_dir}/*/summary.json')):
    run = os.path.basename(os.path.dirname(f))
    s = json.load(open(f))
    df = pd.read_csv(f'{os.path.dirname(f)}/results.csv')
    frames[run] = df
    own_test = df[(df.split == 'own') & (df.kind == 'test')]
    held = df[df.split == 'held_out_doc']
    gen = df[df.split == 'general']
    rows.append({'run': run, 'lambda': s['args']['contrast_weight'], 'numbers': s['numbers_per_document'],
                 'train_right': s['train_right'], 'test_right': s['test_right'], 'test_n': len(own_test),
                 'two_hop_right': s['two_hop_right'], 'test_right_no_intervention': s['test_right_unmodified'],
                 'held_out_docs_give_own_answer': s['held_out_docs_gives_own_answer'],
                 'general_right': s['general_right'], 'general_right_no_intervention': s['general_right_unmodified'],
                 'mean_steps': s['mean_steps']})
    per_doc.append(own_test.groupby('doc_id').modified_right.mean().rename(run))
print("Main table")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nTest accuracy per document")
print(pd.concat(per_doc, axis=1).round(2).to_string())

print("\nTwo-step questions: is the bridge fact one of the document's trained facts?")
n = trained = 0
for doc_id, q in filtered.items():
    answers = {f['answer'] for f in q['facts']}
    evidence = {f['evidence'] for f in q['facts']}
    for h in q['two_hop']:
        n += 1
        same_sentence = h['evidence'] in evidence
        trained += same_sentence
        print(f"  {h['q_id']}: bridge sentence {'is' if same_sentence else 'is not'} a trained fact's evidence "
              f"(trained answers there: {sorted(answers) if same_sentence else '-'})")
print(f"  {trained} of {n} share a sentence with a trained fact; in none is the bridge entity itself a trained answer.")

print("\nWrong test answers, lambda = 1 (first 8 per run)")
for run in ['layers_head_layers_w1', 'kv_top10_4slots_w1', 'heads_last_w1']:
    df = frames[run]
    wrong = df[(df.split == 'own') & (df.kind == 'test') & ~df.modified_right]
    print(f"--- {run}: {len(wrong)} wrong")
    print(wrong[['answers', 'modified']].head(8).to_string(index=False))
