"""
Tables for the wiki2026 coverage experiment (experiment 15): per-document interventions trained on questions, on the
document text, or both, tested on trained facts in new wording, on held-out facts (never in training) and on two-step
questions whose bridge is a trained fact.

Usage (from the function_vectors root):
python findings/2026-11-20_qwen3-4b_wiki2026_coverage/analysis/analyse_wiki2026_coverage.py findings/2026-11-20_qwen3-4b_wiki2026_coverage/outputs
"""
import glob, json, os, re, sys

import pandas as pd

out_dir = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 80)
filtered = {q['doc_id']: q for q in json.load(open('dataset_files/knowledge/wiki2026/filtered_v2.json'))}
ORDER = ['qa_layers8_last', 'qa_layers8_all', 'qa_kv', 'qa_doc_layers8_all', 'doc_layers8_all', 'doc_kv']

rows, frames, logs = [], {}, {}
for run in ORDER:
    s = json.load(open(f'{out_dir}/{run}/summary.json'))
    df = frames[run] = pd.read_csv(f'{out_dir}/{run}/results.csv')
    logs[run] = re.findall(r'(d\d+): (\d+) steps, final own loss ([\d.]+)', open(f'{out_dir}/{run}/run.log').read())
    own = df[df.split == 'own']
    a = s['args']
    rows.append({'run': run, 'mode': a['mode'], 'position': a['position'], 'qa_w': a['qa_weight'],
                 'doc_w': a['doc_lm_weight'], 'numbers': s['numbers_per_document'],
                 'train': s['train_right'], 'test': s['test_right'], 'n_test': (own.kind == 'test').sum(),
                 'held_out_fact': s['held_out_fact_right'], 'n_held': (own.kind == 'held_out_fact').sum(),
                 'two_hop': f"{int(own[own.kind == 'two_hop'].modified_right.sum())}/{(own.kind == 'two_hop').sum()}",
                 'leak': s['held_out_docs_gives_own_answer'], 'general': s['general_right'],
                 'general_unchanged': s['general_unchanged'], 'mean_steps': s['mean_steps'],
                 'final_own_loss_max': s['final_own_loss_max']})
print("Main table (contrast weight 1 everywhere)")
print(pd.DataFrame(rows).round(3).to_string(index=False))
base = frames['qa_layers8_last']
own = base[base.split == 'own']
print(f"\nNo intervention: test {own[own.kind == 'test'].unmodified_right.mean():.3f}, "
      f"held-out facts {own[own.kind == 'held_out_fact'].unmodified_right.mean():.3f}, "
      f"two-step {int(own[own.kind == 'two_hop'].unmodified_right.sum())}/{(own.kind == 'two_hop').sum()}, "
      f"general {base[base.split == 'general'].unmodified_right.mean():.3f}")

print("\nSteps and final own loss per document (early stopping)")
print(pd.DataFrame({run: {d: f"{st} / {float(l):.2f}" for d, st, l in logs[run]} for run in ORDER}).to_string())

print("\nHeld-out facts answered right (by run)")
for run in ORDER:
    df = frames[run]
    right = df[(df.split == 'own') & (df.kind == 'held_out_fact') & df.modified_right]
    if len(right):
        print(f"--- {run}: {len(right)} of 90 phrasings, {right.item_id.str[:8].nunique()} facts")
        print(right[['item_id', 'answers', 'modified']].to_string(index=False))

print("\nTwo-step questions: right? and is the bridge fact's test phrasing right in the same run?")
bridge = {h['q_id']: h['bridge_fact_id'] for q in filtered.values() for h in q['two_hop']}
tab = {}
for run in ORDER:
    df = frames[run]
    own = df[df.split == 'own']
    test_right = own[own.kind == 'test'].set_index(own[own.kind == 'test'].item_id.str[:8]).modified_right
    for _, r in own[own.kind == 'two_hop'].iterrows():
        b = bridge[r.item_id]
        tab.setdefault(r.item_id, {})[run] = f"{'Y' if r.modified_right else '-'} (bridge {'Y' if test_right.get(b, False) else '-'})"
print(pd.DataFrame(tab).T.to_string())
df = frames['qa_layers8_all']
print("\nTwo-step answers, qa_layers8_all")
print(df[(df.split == 'own') & (df.kind == 'two_hop')][['item_id', 'question', 'answers', 'modified']].to_string(index=False))

print("\nTest accuracy per document")
print(pd.concat([frames[r][(frames[r].split == 'own') & (frames[r].kind == 'test')].groupby('doc_id').modified_right
                 .mean().rename(r) for r in ORDER], axis=1).round(2).to_string())

print("\nGeneral questions answered wrong, qa_doc_layers8_all and doc_layers8_all (first 6)")
for run in ['qa_doc_layers8_all', 'doc_layers8_all']:
    df = frames[run]
    g = df[(df.split == 'general') & ~df.modified_right]
    print(f"--- {run}: {len(g)} of {(df.split == 'general').sum()} wrong")
    print(g[['question', 'answers', 'modified']].head(6).to_string(index=False))

print("\nDocument-text loss (mean of first and last 10 steps) against accuracy, text-only runs (outputs/loss_curves.csv)")
curves = pd.read_csv(f'{out_dir}/loss_curves.csv')
for run in ['doc_layers8_all', 'doc_kv']:
    g = curves[curves.run == run].groupby('doc_id').own_loss
    own = frames[run][frames[run].split == 'own']
    t = own.groupby(['doc_id', 'kind']).modified_right.mean().unstack()
    t.insert(0, 'text_loss_last10', g.apply(lambda s: s.tail(10).mean()))
    t.insert(0, 'text_loss_first10', g.apply(lambda s: s.head(10).mean()))
    print(f"--- {run}")
    print(t.round(2).sort_values('text_loss_last10').to_string())
    memorised = t.index[t.text_loss_last10 < 0.15]
    if len(memorised):
        m = own[own.doc_id.isin(memorised)].groupby('kind').modified_right.agg(['sum', 'count'])
        print(f"Documents with final text loss < 0.15 ({', '.join(memorised)}): right / asked")
        print(m.to_string())
