"""
Checks whether the head ranking is driven by a few prompts.

Usage (from the function_vectors root): python findings/<experiment>/analysis/inspect_indirect_effect.py findings/<experiment>/outputs
"""
import json, sys

import numpy as np
import torch

out_dir = sys.argv[1]
summary = json.load(open(f"{out_dir}/summary.json"))
data = torch.load(f"{out_dir}/fv_extraction.pt")
ie = data["indirect_effect"]  # trials x layers x heads
n_trials, n_layers, n_heads = ie.shape
trials_per_person = n_trials // len(summary["fv_norms"])

for name, sanity in summary["sanity"].items():
    print(name, sanity)

sim = np.array(summary["fv_cosine_similarity"]["matrix"])
off_diagonal = ~np.eye(len(sim), dtype=bool)
print("\nFV cosine similarity:\n", np.round(sim, 3))
print(f"off-diagonal mean {sim[off_diagonal].mean():.3f}, min {sim[off_diagonal].min():.3f}, max {sim[off_diagonal].max():.3f}")

print("\nshape", tuple(ie.shape))
print("max per trial", [f"{x:.3e}" for x in ie.flatten(1).max(1).values.tolist()])
print("distinct values per trial", [len(t.unique()) for t in ie])

mean_ie = ie.mean(0)
values, indices = mean_ie.flatten().topk(10)
print("top 10 heads", [(i // n_heads, i % n_heads, f"{v:.3e}") for v, i in zip(values.tolist(), indices.tolist())])
top_per_trial = ie.flatten(1)[:, indices]
share = top_per_trial.sum(1) / top_per_trial.sum()
print("share of top-10 effect per trial", [f"{s:.2f}" for s in share.tolist()])
print(f"largest single-trial share: {share.max():.2f}")

per_person = [ie[k:k + trials_per_person].mean(0) for k in range(0, n_trials, trials_per_person)]
top_sets = [set(p.flatten().topk(10).indices.tolist()) for p in per_person]
print("pairwise top-10 head overlap between people", [[len(a & b) for b in top_sets] for a in top_sets])
