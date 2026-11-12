"""
Checks whether an injected FV answers each question or pushes the same city for every question.

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_injection_smoke/analysis/analyse_injection.py findings/2026-11-12_qwen3-4b_injection_smoke/outputs
"""
import json, sys

import pandas as pd

out_dir = sys.argv[1]
df = pd.read_csv(f"{out_dir}/injection_results.csv")
summary = json.load(open(f"{out_dir}/summary.json"))
pd.set_option('display.width', 200); pd.set_option('display.max_columns', None)

own = df[df.condition == 'own']
print("Own FV: distinct predicted cities per person (4 questions each), averaged over people")
print(own.groupby(['layer', 'scale', 'person']).pred_city.nunique().groupby(['layer', 'scale']).mean().unstack().round(2))

print("\nOwn FV: share of questions where the prediction is one of the person's own cities")
print(own.groupby(['layer', 'scale']).owner_city_hit.mean().unstack().round(2))

layer, scale = summary['configs']['best_own']
for condition in ['own', 'same_cities_partner', 'different_cities_person']:
    sub = df[(df.condition == condition) & (df.layer == layer) & (df.scale == scale)]
    print(f"\n{condition} at layer {layer}, scale {scale}: predicted city per question")
    print(sub.pivot(index='person', columns='relation', values='pred_city')[['born_city', 'work_city', 'study_city', 'home_city']])

# Which of the owner's cities does the FV push? Map the predicted city back to the fact it belongs to.
people = json.load(open('dataset_files/knowledge/fictional_people_smoke.json'))['people']
city_to_relation = {p['name']: {c: r for r, c in p['facts'].items()} for p in people}
sub = df[(df.condition == 'own') & (df.layer == layer) & (df.scale == scale)]
print(f"\nOwn FV at layer {layer}, scale {scale}: which of the person's facts the predicted city belongs to")
print(sub.assign(pushed_fact=[city_to_relation[p].get(c) for p, c in zip(sub.person, sub.pred_city)])
         .groupby('person').pushed_fact.agg(lambda s: ', '.join(sorted(set(s.dropna())))))

partner = df[(df.condition == 'same_cities_partner') & (df.layer == layer) & (df.scale == scale)]
merged = sub.merge(partner, on=['person', 'relation'], suffixes=('_own', '_partner'))
print(f"\nSame question, own FV vs same-cities partner FV: same predicted city in {(merged.pred_city_own == merged.pred_city_partner).mean():.2f} of questions")
print("\nHidden-state norm at the last token by layer:", [round(x) for x in summary['hidden_state_norms_last_token']])
print("FV norms:", {k: round(v, 1) for k, v in summary['fv_norms'].items()})
