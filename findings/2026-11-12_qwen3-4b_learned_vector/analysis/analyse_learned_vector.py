"""
Where do learned vectors fail on held-out paraphrases, and what do they answer for the twin's questions?

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_learned_vector/analysis/analyse_learned_vector.py findings/2026-11-12_qwen3-4b_learned_vector/outputs
"""
import json, sys

import pandas as pd

out_dir = sys.argv[1]
df = pd.read_csv(f"{out_dir}/learned_vector_results.csv")
summary = json.load(open(f"{out_dir}/summary.json"))
people = json.load(open('dataset_files/knowledge/fictional_people_smoke.json'))['people']
own_cities = {p['name']: set(p['facts'].values()) for p in people}
pd.set_option('display.width', 200); pd.set_option('display.max_columns', None)

learned = df[df.vector == 'learned']
test = learned[learned.split == 'test'].copy()
print("Held-out accuracy by question type (rows: position, layer):")
print(test.groupby(['position', 'layer', 'relation']).correct_city.mean().unstack()[['born_city', 'work_city', 'study_city', 'home_city']].round(2))

wrong = test[~test.correct_city].copy()
wrong['error'] = ['another of the person\'s cities' if c in own_cities[o] else 'a city not the person\'s'
                  for c, o in zip(wrong.pred_city, wrong.vector_owner)]
print("\nHeld-out errors by type (all positions and layers):")
print(wrong.error.value_counts())
print("Held-out questions that fail most often (out of 8 position-layer configurations):")
print(wrong.groupby(['vector_owner', 'relation', 'question']).size().sort_values(ascending=False).head(8))

twin = learned[learned.split == 'twin_test'].copy()
twin['answer_given'] = ['the twin\'s answer (follows the name)' if c else ('the vector owner\'s answer for that question type' if o else 'something else')
                        for c, o in zip(twin.correct_city, twin.owner_answer_hit)]
print("\nVector applied to the twin's held-out questions:")
print(twin.groupby(['position', 'layer']).answer_given.value_counts(normalize=True).unstack().fillna(0).round(2))

print("\nVector norms by position and layer (range over people):")
norms = pd.Series(summary['vector_norms'])
idx = norms.index.str.split(' / ')
print(pd.DataFrame({'position': idx.str[0], 'layer': idx.str[1], 'norm': norms.values}).groupby(['position', 'layer']).norm.agg(['min', 'max']).round(1))
