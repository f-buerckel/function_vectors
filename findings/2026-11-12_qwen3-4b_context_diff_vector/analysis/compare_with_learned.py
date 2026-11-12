"""
Why does the context-difference vector fail? Compares the vectors between people (generic vs person-specific),
with the learned vectors at the same position and layer, and with the hidden-state norm.

Usage (from the function_vectors root):
python findings/2026-11-12_qwen3-4b_context_diff_vector/analysis/compare_with_learned.py \
    findings/2026-11-12_qwen3-4b_context_diff_vector/outputs findings/2026-11-12_qwen3-4b_learned_vector/outputs
"""
import sys

import torch

diff_dir, learned_dir = sys.argv[1], sys.argv[2]
diff = torch.load(f"{diff_dir}/context_diff_vectors.pt")  # 'name / position' -> layers x hidden
learned = torch.load(f"{learned_dir}/learned_vectors.pt")['vectors']  # 'position / layer L / name' -> hidden
names = sorted({k.split(' / ')[0] for k in diff})
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0).item()

for position in ['name', 'last']:
    print(f"\n=== position: {position}")
    for layer in [6, 12, 18, 24]:
        d = {n: diff[f'{n} / {position}'][layer] for n in names}
        mean = torch.stack(list(d.values())).mean(0)
        between = [cos(d[a], d[b]) for i, a in enumerate(names) for b in names[i + 1:]]
        between_specific = [cos(d[a] - mean, d[b] - mean) for i, a in enumerate(names) for b in names[i + 1:]]
        share_shared = mean.norm().item() ** 2 / torch.stack([v.norm() ** 2 for v in d.values()]).mean().item()
        with_learned = [cos(d[n], learned[f'{position} / layer {layer} / {n}']) for n in names]
        with_learned_specific = [cos(d[n] - mean, learned[f'{position} / layer {layer} / {n}']) for n in names]
        print(f"layer {layer:2d}: diff norm {torch.stack([v.norm() for v in d.values()]).mean():6.1f} | "
              f"cosine between people {sum(between)/len(between):.3f} | share of squared norm shared by all people {share_shared:.2f} | "
              f"cosine with learned vector {sum(with_learned)/len(with_learned):+.3f} "
              f"(person-specific part: {sum(with_learned_specific)/len(with_learned_specific):+.3f})")
