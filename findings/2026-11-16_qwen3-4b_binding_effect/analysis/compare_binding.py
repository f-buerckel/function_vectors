"""
Compares FVs from heads ranked by the binding effect with FVs from heads ranked by the log-probability effect
(experiments 2 and 3), on the same injection test.

Usage (from the function_vectors root):
python findings/2026-11-16_qwen3-4b_binding_effect/analysis/compare_binding.py \
    findings/2026-11-16_qwen3-4b_binding_effect/outputs findings/2026-11-12_qwen3-4b_injection_smoke/outputs \
    findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs
"""
import json, sys

import pandas as pd

binding_dir, logprob_injection_dir, logprob_extraction_dir = sys.argv[1:4]
pd.set_option('display.width', 220); pd.set_option('display.max_columns', None)

new = json.load(open(f"{binding_dir}/extraction/summary.json"))
old = json.load(open(f"{logprob_extraction_dir}/summary.json"))
new_heads = [(l, h) for l, h, _ in new['top_heads']]
old_heads = [(l, h) for l, h, _ in old['top_heads']]
print("Top-10 heads, binding effect:", [(l, h, round(s, 3)) for l, h, s in new['top_heads']])
print("Top-10 heads, log-prob effect:", old_heads)
print("Shared heads:", sorted(set(new_heads) & set(old_heads)))
print("Ranking stability (binding):", new['ranking_stability'])
names = new['fv_cosine_similarity']['names']
sim = pd.DataFrame(new['fv_cosine_similarity']['matrix'], index=names, columns=names)
print("\nFV cosine similarity (binding):\n", sim.round(2))

frames = {'binding': pd.read_csv(f"{binding_dir}/injection/injection_results.csv"),
          'logprob': pd.read_csv(f"{logprob_injection_dir}/injection_results.csv")}
for label, df in frames.items():
    print(f"\n===== {label}: accuracy (closed set) by condition, layer and scale")
    print(df.groupby(['condition', 'layer', 'scale']).correct.mean().unstack().round(2).to_string())
    own = df[df.condition == 'own']
    print(f"{label}: own FV, distinct predicted cities per person (4 questions), mean over people")
    print(own.groupby(['layer', 'scale', 'person']).pred_city.nunique().groupby(['layer', 'scale']).mean().unstack().round(2).to_string())
    print(f"{label}: own FV, share of predictions that are one of the person's cities")
    print(own.groupby(['layer', 'scale']).owner_city_hit.mean().unstack().round(2).to_string())

print("\nTwin test: own minus same-cities partner accuracy, averaged over all layers and scales >= 1")
for label, df in frames.items():
    d = df[df.scale >= 1].groupby(['condition']).correct.mean()
    print(f"{label}: own {d['own']:.3f}, partner {d['same_cities_partner']:.3f}, different-cities person {d['different_cities_person']:.3f}, random {d['random']:.3f}")
    best = df[df.condition == 'own'].groupby(['layer', 'scale']).correct.mean()
    print(f"{label}: best own cell {best.idxmax()} = {best.max():.3f}; cells above 0.25: {(best > 0.25).sum()} of {len(best)}")
