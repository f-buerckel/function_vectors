"""
Compares this run with the 2026-11-06 run (shuffled corruption, probability effect):
head ranking stability, overlap of top heads, and FV similarity between people who share the same cities.

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/analysis/compare_runs.py \
    findings/2026-11-06_qwen3-4b_extraction_smoke/outputs findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs
"""
import json, sys

import numpy as np
import torch

old_dir, new_dir = sys.argv[1], sys.argv[2]
old, new = (json.load(open(f"{d}/summary.json")) for d in (old_dir, new_dir))
ie = torch.load(f"{new_dir}/fv_extraction.pt")["indirect_effect"]  # trials x layers x heads
n_people = len(new["fv_norms"])
trials_per_person = ie.shape[0] // n_people

print("sanity (with context, without context):", {k: (v["with_context_top1"], v["no_context_top1"]) for k, v in new["sanity"].items()})
print("ranking stability:", {k: v for k, v in new["ranking_stability"].items() if k != "trial_share_of_top_effect"})
print("share of top-10 effect per prompt:", [round(x, 2) for x in new["ranking_stability"]["trial_share_of_top_effect"]])

old_heads = {(l, h) for l, h, _ in old["top_heads"]}
new_heads = {(l, h) for l, h, _ in new["top_heads"]}
print("top heads, this run:", sorted(new_heads))
print("shared with 6 Nov run:", len(old_heads & new_heads), sorted(old_heads & new_heads))

mean_ie = ie.mean(0)
top = mean_ie.flatten().topk(10).indices
per_person = [ie[k:k + trials_per_person].mean(0) for k in range(0, ie.shape[0], trials_per_person)]
top_sets = [set(p.flatten().topk(10).indices.tolist()) for p in per_person]
print("per-person top-10 overlap:", [[len(a & b) for b in top_sets] for a in top_sets])
print("each person's mean effect on the shared top-10 heads:", [round(p.flatten()[top].mean().item(), 3) for p in per_person])
full = set(top.tolist())
leave_one_out = [len(full & set(torch.cat([ie[:k], ie[k + trials_per_person:]]).mean(0).flatten().topk(10).indices.tolist()))
                 for k in range(0, ie.shape[0], trials_per_person)]
print("leave-one-person-out top-10 overlap with full ranking:", leave_one_out)
print("positive effect summed per layer:", [round(x, 2) for x in mean_ie.clamp(min=0).sum(1).tolist()])

# Same-city pairs are specific to fictional_people_smoke.json
sim = np.array(new["fv_cosine_similarity"]["matrix"])
same = [(0, 3), (1, 4), (2, 5)]
different = [(i, j) for i in range(n_people) for j in range(i + 1, n_people) if (i, j) not in same]
print("FV cosine, same-city pairs:", [round(sim[i, j], 3) for i, j in same])
print("FV cosine, different cities: mean %.3f, range %.3f-%.3f" % (np.mean([sim[i, j] for i, j in different]),
      min(sim[i, j] for i, j in different), max(sim[i, j] for i, j in different)))
