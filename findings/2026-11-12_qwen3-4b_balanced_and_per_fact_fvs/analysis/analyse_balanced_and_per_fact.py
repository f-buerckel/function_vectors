"""
Analyses the balanced and per-fact FV injection runs.

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_balanced_and_per_fact_fvs/analysis/analyse_balanced_and_per_fact.py \
    findings/2026-11-12_qwen3-4b_balanced_and_per_fact_fvs/outputs findings/2026-11-12_qwen3-4b_injection_smoke/outputs
"""
import json, sys

import pandas as pd

out_dir, unbalanced_dir = sys.argv[1], sys.argv[2]
pd.set_option('display.width', 200); pd.set_option('display.max_columns', None)
people = json.load(open('dataset_files/knowledge/fictional_people_smoke.json'))['people']
city_to_relation = {p['name']: {c: r for r, c in p['facts'].items()} for p in people}
relations = ['born_city', 'work_city', 'study_city', 'home_city']

balanced = pd.read_csv(f"{out_dir}/injection_balanced/injection_results.csv")
unbalanced = pd.read_csv(f"{unbalanced_dir}/injection_results.csv")
per_fact = pd.read_csv(f"{out_dir}/injection_per_fact/injection_results.csv")

print("=== Balanced FVs")
own = balanced[balanced.condition == 'own']
print("Distinct predicted cities per person (4 questions), averaged over people:")
print(own.groupby(['layer', 'scale', 'person']).pred_city.nunique().groupby(['layer', 'scale']).mean().unstack().round(2))
print("Share of predictions that are one of the person's own cities:")
print(own.groupby(['layer', 'scale']).owner_city_hit.mean().unstack().round(2))
for layer, scale in [(15, 2.0), (27, 8.0)]:
    print(f"\nPredicted city per question, layer {layer}, scale {scale}: unbalanced vs balanced own FV")
    for name, df in [('unbalanced', unbalanced), ('balanced', balanced)]:
        sub = df[(df.condition == 'own') & (df.layer == layer) & (df.scale == scale)]
        print(name); print(sub.pivot(index='person', columns='relation', values='pred_city')[relations])
    sub = balanced[(balanced.condition == 'own') & (balanced.layer == layer) & (balanced.scale == scale)]
    partner = balanced[(balanced.condition == 'same_cities_partner') & (balanced.layer == layer) & (balanced.scale == scale)]
    merged = sub.merge(partner, on=['person', 'relation'], suffixes=('_own', '_partner'))
    print(f"balanced: own and same-cities partner FV give the same city in {(merged.pred_city_own == merged.pred_city_partner).mean():.2f} of questions")

print("\n=== Per-fact FVs")
for condition in ['own', 'other_fact', 'different_cities_person', 'random']:
    sub = per_fact[per_fact.condition == condition]
    print(f"\n{condition}: right city (closed set)")
    print(sub.groupby(['layer', 'scale']).correct.mean().unstack().round(2))
    if condition in ['other_fact', 'different_cities_person']:
        print(f"{condition}: prediction is the city the vector was built from")
        print(sub.groupby(['layer', 'scale']).owner_answer_hit.mean().unstack().round(2))
print("\nown: first token over the full vocabulary is the right city")
print(per_fact[per_fact.condition == 'own'].groupby(['layer', 'scale']).full_vocab_correct.mean().unstack().round(2))

summary = json.load(open(f"{out_dir}/fact_fvs/summary.json"))
print("\nPer-fact FV cosine similarity:")
for group, stats in summary['per_fact_fv_similarity'].items():
    print(f"  {group}: mean {stats['mean']:.3f}, range {stats['min']:.3f} to {stats['max']:.3f} ({stats['n_pairs']} pairs)")
print("Balanced FV vs mean of per-fact FVs, largest relative difference:", summary['balanced_equals_mean_of_per_fact_relative_error'])
