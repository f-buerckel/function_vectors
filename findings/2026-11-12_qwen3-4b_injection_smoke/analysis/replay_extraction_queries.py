"""
Replays the random draws of the extraction run (commit e20deb4, seed 42, --effect logprob --corruption other_person,
--n_mean_trials 10 --n_ie_trials 4) without the model, and counts which fact each mean-activation prompt asked about.

It mirrors the order of np.random calls in extract_knowledge_fv.py and the repo functions it calls. It was not checked
against the actual prompts, so treat the counts as a reconstruction.

Usage (from the function_vectors root): python findings/2026-11-12_qwen3-4b_injection_smoke/analysis/replay_extraction_queries.py
"""
import json, random
from collections import Counter

import numpy as np

people = json.load(open("dataset_files/knowledge/fictional_people_smoke.json"))["people"]
relations = list(people[0]["facts"])
n_facts = len(relations)
n_queries = 2 * n_facts  # two extraction paraphrases per fact, ordered by fact

random.seed(42); np.random.seed(42)
for person in people:  # sanity check with context: one draw of demonstrations per held-out question
    for _ in range(n_facts):
        np.random.choice(n_facts, n_facts, replace=False)

for person in people:
    asked = Counter()
    for _ in range(10):  # get_mean_head_activations: demonstrations, then the query
        np.random.choice(n_facts, n_facts, replace=False)
        asked[relations[np.random.choice(np.arange(n_queries), 1, replace=False)[0] // 2]] += 1
    cities = set(person["facts"].values())
    n_other = sum(1 for other in people if not cities & set(other["facts"].values()))
    for _ in range(4):  # compute_indirect_effect with other-person corruption
        np.random.randint(n_other)
        np.random.choice(n_facts, n_facts, replace=False)
        np.random.choice(np.arange(n_queries), 1, replace=False)
    print(f"{person['name']:16s} {dict(sorted(asked.items()))}")
