"""
Compares learned interventions over heads or layers with the single learned vector (8 facts per person).

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_learned_heads_and_layers/analysis/analyse_heads_and_layers.py \
    findings/2026-11-12_qwen3-4b_learned_heads_and_layers/outputs findings/2026-11-12_qwen3-4b_learned_vector_capacity/outputs
"""
import json, os, sys

import pandas as pd

out_dir, single_dir = sys.argv[1], sys.argv[2]
pd.set_option('display.width', 220); pd.set_option('display.max_columns', None); pd.set_option('display.max_colwidth', 60)
people = json.load(open('dataset_files/knowledge/fictional_people_capacity.json'))['people']
all_cities = {c for p in people for c in p['facts'].values()}
own = {p['name']: set(p['facts'].values()) for p in people}

rows, frames = [], []
for name in sorted(os.listdir(out_dir)):
    path = f"{out_dir}/{name}/summary.json"
    if not os.path.exists(path):
        continue
    s = json.load(open(path))
    rows.append({'intervention': name, 'numbers': s['numbers_per_person'], 'train': s['train_accuracy'],
                 'held_out': s['held_out_accuracy'], 'all_held_out_right': s['people_with_all_held_out_right'],
                 'known_facts_right': s['known_facts_still_right'], 'max_final_loss': s['final_train_loss_max']})
    df = pd.read_csv(f"{out_dir}/{name}/results.csv")
    df['intervention'] = name
    frames.append(df)

single = pd.read_csv(f"{single_dir}/facts_8/learned_vector_results.csv")
for (position, layer), g in single[single.vector == 'learned'].groupby(['position', 'layer']):
    rows.append({'intervention': f'single_vector_{position}_layer{layer}', 'numbers': 2560,
                 'train': g[g.split == 'train'].correct_city.mean(), 'held_out': g[g.split == 'test'].correct_city.mean(),
                 'all_held_out_right': (g[g.split == 'test'].groupby('vector_owner').correct_city.mean() == 1).mean(),
                 'known_facts_right': g[g.split == 'known_facts'].correct_token.mean(), 'max_final_loss': None})
print(pd.DataFrame(rows).round(3).to_string(index=False))

df = pd.concat(frames)
order = list(people[0]['facts'])
print("\nHeld-out accuracy by fact type:")
print(df[df.split == 'test'].groupby(['intervention', 'relation']).correct_city.mean().unstack()[order].round(2))

known = df[df.split == 'known_facts'].copy()
known['answer_given'] = ['right capital' if c else ("one of the person's cities" if t in own[p] else
                         ('another dataset city' if t in all_cities else 'something else'))
                         for c, t, p in zip(known.correct_token, known.pred_token, known.person)]
print("\nWhat the interventions answer to the capital questions:")
print(known.groupby('intervention').answer_given.value_counts(normalize=True).unstack().fillna(0).round(2))
print("\nExamples:")
print(known[known.person == people[0]['name']][['intervention', 'question', 'pred_token']].to_string(index=False))
