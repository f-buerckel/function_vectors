"""
Tables for experiment 13: binding vs log-probability head selection with the paper's prompt counts, FVs from heads
below layer 24 only, and the check of the --fast / --early_stop training options against experiment 10.

Usage (from the function_vectors root):
python findings/2026-11-16_qwen3-4b_binding_paper_counts/analysis/analyse_paper_counts.py findings/2026-11-16_qwen3-4b_binding_paper_counts/outputs
"""
import json, sys

import pandas as pd

out = sys.argv[1]
pd.set_option('display.width', 220)


def injection_summary(path):
    df = pd.read_csv(path)
    d = df[df.scale >= 1].groupby('condition').correct.mean()
    own = df[df.condition == 'own']
    best = own.groupby(['layer', 'scale']).correct.mean()
    big = own[own.scale >= 4]
    return {'own': d['own'], 'partner': d['same_cities_partner'], 'different_person': d['different_cities_person'],
            'random': d['random'], 'best_cell': f"layer {best.idxmax()[0]}, scale {best.idxmax()[1]}", 'best': best.max(),
            'cells_above_0.25': int((best > 0.25).sum()), 'owner_city_at_scale_4_8': big.owner_city_hit.mean(),
            'distinct_cities_at_scale_4_8': big.groupby(['layer', 'scale', 'person']).pred_city.nunique().mean()}


rows = []
for effect in ['binding', 'logprob']:
    s = json.load(open(f'{out}/{effect}/extraction/summary.json'))
    st = s['ranking_stability']
    print(f"{effect}, all layers: top heads {[(l, h) for l, h, _ in s['top_heads']]}")
    print(f"  split-half overlap {st['split_half_top_overlap']}/10, largest single-prompt share {st['max_single_trial_share_of_top_effect']:.3f}")
    sim = s['fv_cosine_similarity']
    names = sim['names']
    twins = [('Elara Voss', 'Davor Quill'), ('Tomas Wrenfield', 'Mira Solbrandt'), ('Ines Kallow', 'Joren Thale')]
    print("  twin FV cosines:", {f"{a}/{b}": round(sim['matrix'][names.index(a)][names.index(b)], 2) for a, b in twins})
    rows.append({'heads': f'{effect}, all layers', **injection_summary(f'{out}/{effect}/injection/injection_results.csv')})
    b24 = json.load(open(f'{out}/below24/fv_{effect}_below24.json'))
    print(f"{effect}, below layer 24: top heads {[(l, h) for l, h, _ in b24['top_heads']]}")
    rows.append({'heads': f'{effect}, below layer 24',
                 **injection_summary(f'{out}/below24/injection_{effect}_below24/injection_results.csv')})
print("\nInjection (accuracy averaged over all layers and scales >= 1)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nSpeed-up check against experiment 10 (contrast weight 1)")
ref = {'heads_last_w1_fast': 'heads_last_w1', 'heads_last_w1_fast_stop': 'heads_last_w1',
       'kv_top10_4slots_w1_fast_stop': 'kv_top10_4slots_w1',
       'layers_head_layers_last_w1_fast_stop': 'layers_head_layers_last_w1',
       'vector_layer12_name_w1_fast_stop': 'vector_layer12_name_w1'}
times = dict(line.split()[:2] for line in open(f'{out}/fast_check/timings.txt'))
keys = ['own_held_out_right', 'capitals_right', 'other_people_answered_with_owner_city', 'other_people_mean_kl']
rows = []
for new, old in ref.items():
    a = json.load(open(f'findings/2026-11-15_qwen3-4b_contrast_training/outputs/{old}/summary.json'))
    b = json.load(open(f'{out}/fast_check/{new}/summary.json'))
    steps = [int(l.split(': ')[1].split()[0]) for l in open(f'{out}/fast_check/{new}/run.log') if l.endswith(' steps\n')]
    rows.append({'run': new, 'seconds': int(times[new]), 'mean_steps': sum(steps) / len(steps) if steps else None,
                 **{f'{k} (exp 10 -> now)': f"{a[k]:.3f} -> {b[k]:.3f}" for k in keys}})
print(pd.DataFrame(rows).to_string(index=False))
