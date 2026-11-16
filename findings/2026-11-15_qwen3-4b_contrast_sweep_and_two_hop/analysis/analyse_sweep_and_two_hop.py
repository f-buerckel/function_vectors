"""
Tables for the contrast-weight sweep and the generation-based evaluation (two-hop questions, side effects, memory attention).

Usage (from the function_vectors root):
python findings/2026-11-15_qwen3-4b_contrast_sweep_and_two_hop/analysis/analyse_sweep_and_two_hop.py \
    findings/2026-11-15_qwen3-4b_contrast_sweep_and_two_hop/outputs findings/2026-11-15_qwen3-4b_contrast_training/outputs
"""
import glob, json, os, sys

import pandas as pd

sweep_dir, main_dir = sys.argv[1], sys.argv[2]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
people = json.load(open('dataset_files/knowledge/fictional_people_capacity.json'))['people']
COUNTRY = {'Athens': 'Greece', 'Boston': 'US', 'Cairo': 'Egypt', 'Chicago': 'US', 'Dublin': 'Ireland', 'Lima': 'Peru',
           'Lisbon': 'Portugal', 'Madrid': 'Spain', 'Mumbai': 'India', 'Nairobi': 'Kenya', 'Oslo': 'Norway',
           'Prague': 'Czech', 'Seoul': 'Korea', 'Sydney': 'Australia', 'Toronto': 'Canada', 'Vienna': 'Austria'}
ALIASES = {'US': ['United States', 'USA', 'U.S.']}
own_countries = {p['name']: [COUNTRY[c] for c in p['facts'].values()] for p in people}
has_country = lambda text, cs: any(a.lower() in str(text).lower() for c in cs for a in ALIASES.get(c, [c]))


def run_label(path):
    return ('sweep/' if 'knowledge_contrast_sweep' in path else '') + os.path.basename(os.path.dirname(path))


# 1. First-token metrics of every training run (sweep and experiment 10)
rows = []
for f in sorted(glob.glob(f'{main_dir}/*/summary.json') + glob.glob(f'{sweep_dir}/sweep/*/summary.json')):
    s = json.load(open(f))
    rows.append({'run': ('sweep/' if '/sweep/' in f else '') + os.path.basename(os.path.dirname(f)),
                 'lambda': s['args']['contrast_weight'], 'seed': s['args']['seed'],
                 'held_out_right': s['own_held_out_right'], 'capitals_right': s['capitals_right'],
                 'others_owner_city': s['other_people_answered_with_owner_city'],
                 'others_unchanged': s['other_people_unchanged'], 'others_kl': s['other_people_mean_kl'],
                 'capitals_kl': s['capitals_mean_kl']})
print("First-token metrics (from training runs)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 2. Generation-based evaluation
bag = sum(sum(cs.count(c) for c in cs) / 64 for cs in own_countries.values()) / len(own_countries)
print(f"\nTwo-hop bag-of-answers baseline (country of a random one of the person's cities): {bag:.3f}")
rows = []
for f in sorted(glob.glob(f'{sweep_dir}/generation_eval/knowledge_contrast*/*/generation_eval.csv')):
    df = pd.read_csv(f)
    g = lambda s: df[df.set == s]
    oth2 = g('other_two_hop')
    rows.append({'run': run_label(f),
                 'held_out_right': g('own_held_out').modified_right.mean(),
                 'two_hop_right': g('own_two_hop').modified_right.mean(),
                 'two_hop_in_context': g('own_two_hop').in_context_right.astype(bool).mean(),
                 'two_hop_unmodified': g('own_two_hop').unmodified_right.mean(),
                 'others_owner_city': g('other_held_out').modified_has_owner_city.mean(),
                 'others_two_hop_owner_country': oth2.modified_has_owner_country.mean(),
                 'others_two_hop_owner_country_unmodified': oth2.apply(lambda r: has_country(r.unmodified, own_countries[r.person]), axis=1).mean(),
                 'capitals_right': g('capitals').modified_right.mean(),
                 'general_city_right': g('general_city').modified_right.mean(),
                 'general_city_unmodified': g('general_city').unmodified_right.mean(),
                 'general_other_right': g('general_other').modified_right.mean(),
                 'attention_own': g('own_held_out').memory_attention.mean(),
                 'attention_other_people': g('other_held_out').memory_attention.mean(),
                 'attention_general': pd.concat([g('capitals'), g('general_city'), g('general_other')]).memory_attention.mean()})
print("\nFull greedy answers (16 tokens)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 3. Wrong two-hop answers: another of the person's countries, a bare city, or something else
print("\nWrong own two-hop answers, lambda = 1 runs")
for f in sorted(glob.glob(f'{sweep_dir}/generation_eval/knowledge_contrast/*_w1/generation_eval.csv')):
    df = pd.read_csv(f)
    w = df[(df.set == 'own_two_hop') & ~df.modified_right]
    other_own = w.apply(lambda r: has_country(r.modified, own_countries[r.person]), axis=1)
    print(f"{run_label(f)}: {len(w)} wrong, {other_own.mean():.2f} name another of the person's countries; examples:",
          w.modified.head(4).tolist())
