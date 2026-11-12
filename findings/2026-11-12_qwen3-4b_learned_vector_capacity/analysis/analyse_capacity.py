"""
Capacity of learned vectors: held-out accuracy by number of facts, by fact type, and error types.

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_learned_vector_capacity/analysis/analyse_capacity.py findings/2026-11-12_qwen3-4b_learned_vector_capacity/outputs
"""
import json, sys

import pandas as pd

out_dir = sys.argv[1]
people = json.load(open('dataset_files/knowledge/fictional_people_capacity.json'))['people']
pd.set_option('display.width', 200); pd.set_option('display.max_columns', None)

frames = []
for k in [2, 4, 8]:
    df = pd.read_csv(f"{out_dir}/facts_{k}/learned_vector_results.csv")
    df['n_facts'] = k
    frames.append(df)
    summary = json.load(open(f"{out_dir}/facts_{k}/summary.json"))
    losses = summary['final_train_loss'].values()
    print(f"{k} facts: final training loss max {max(losses):.4f}")
df = pd.concat(frames)
learned = df[(df.vector == 'learned') & (df.split == 'test')].copy()

print("\nHeld-out accuracy (rows: position, layer; columns: facts per vector):")
print(learned.groupby(['position', 'layer', 'n_facts']).correct_city.mean().unstack().round(2))
print("\nNo vector, held-out accuracy:", df[(df.vector == 'none') & (df.split == 'test')].groupby('n_facts').correct_city.mean().round(3).to_dict())

order = list(people[0]['facts'])
print("\nHeld-out accuracy by fact type, averaged over positions and layers (columns: facts per vector):")
print(learned.groupby(['relation', 'n_facts']).correct_city.mean().unstack().reindex(order).round(2))

own = {p['name']: list(p['facts'].values()) for p in people}
wrong = learned[~learned.correct_city].copy()
wrong['error'] = ['a city of a fact in the vector' if c in own[o][:k] else ('one of the person\'s other cities' if c in own[o] else 'a city not the person\'s')
                  for c, o, k in zip(wrong.pred_city, wrong.vector_owner, wrong.n_facts)]
print("\nHeld-out errors by type:")
print(wrong.groupby('n_facts').error.value_counts().unstack().fillna(0).astype(int))

known = df[df.split == 'known_facts'].groupby(['position', 'layer', 'n_facts']).correct_token.mean().unstack().round(2)
print("\nKnown facts (capitals) still right:"); print(known)
