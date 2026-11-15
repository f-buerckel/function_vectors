"""
Compares learned interventions trained without (w0) and with (w1) contrast questions.

Usage (from the function_vectors root):
python findings/2026-11-15_qwen3-4b_contrast_training/analysis/analyse_contrast.py findings/2026-11-15_qwen3-4b_contrast_training/outputs
"""
import json, os, sys

import pandas as pd

out_dir = sys.argv[1]
pd.set_option('display.width', 220); pd.set_option('display.max_columns', None)
people = json.load(open('dataset_files/knowledge/fictional_people_capacity.json'))['people']
all_cities = {c for p in people for c in p['facts'].values()}
own = {p['name']: set(p['facts'].values()) for p in people}

rows, errors = [], []
for name in sorted(os.listdir(out_dir)):
    path = f"{out_dir}/{name}/results.csv"
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    s = json.load(open(f"{out_dir}/{name}/summary.json"))
    test, cap, oth = (df[df.split == x] for x in ['test', 'capitals', 'other_people_held_out'])
    assert (cap.unmodified_token == cap.answer).all(), "unmodified model should answer all capitals right"
    rows.append({'run': name, 'numbers': s['numbers_per_person'],
                 'train_right': df[df.split == 'train'].correct.mean(),
                 'held_out_right': test.correct.mean(),
                 'held_out_first_token_right': test.correct_token.mean(),
                 'people_all_8': int((test.groupby('person').correct.mean() == 1).sum()),
                 'capitals_right': cap.correct_token.mean(),
                 'capitals_wrong_city': (cap.pred_token.isin(all_cities) & (cap.pred_token != cap.answer)).mean(),
                 'capitals_kl': cap.kl_to_unmodified.mean(),
                 'others_unchanged': oth.unchanged.mean(),
                 'others_given_owner_city': oth.gives_owner_city.mean(),
                 'others_given_any_city': oth.pred_token.isin(all_cities).mean(),
                 'others_kl': oth.kl_to_unmodified.mean()})
    wrong = test[~test.correct]
    errors.append({'run': name, 'held_out_errors': len(wrong),
                   'error_is_other_own_city': wrong.apply(lambda r: r.pred_city in own[r.person], axis=1).mean() if len(wrong) else float('nan'),
                   'held_out_first_token_not_a_city': (~test.pred_token.isin(all_cities)).mean()})
    print(f"\n{name}: held-out right by relation", test.groupby('relation').correct.mean().round(2).to_dict())

print("\nMain table")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nHeld-out errors")
print(pd.DataFrame(errors).round(3).to_string(index=False))

print("\nOther people's questions whose first token changed, with contrast (w1): most common (unmodified -> modified)")
for name in sorted(os.listdir(out_dir)):
    if name.endswith('_w1'):
        df = pd.read_csv(f"{out_dir}/{name}/results.csv")
        ch = df[(df.split == 'other_people_held_out') & ~df.unchanged]
        print(f"{name}: {len(ch)} changed;", ch.apply(lambda r: f"{r.unmodified_token!r}->{r.pred_token!r}", axis=1).value_counts().head(4).to_dict())
