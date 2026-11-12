"""
Compares mean last-token head activations between people who share the same set of answers
(swapped roles) and people with different answers, across layers and head sets.

Same-answer pairs are specific to fictional_people_smoke.json: (0, 3), (1, 4), (2, 5).
Usage (from the function_vectors root): python findings/<experiment>/analysis/compare_activation_similarity.py findings/<experiment>/outputs
"""
import json, sys

import torch

out_dir = sys.argv[1]
data = torch.load(f"{out_dir}/fv_extraction.pt")
activations = data["mean_activations_last_token"]
A = torch.stack([activations[n].float() for n in activations])  # people x layers x heads x head_dim
n_people, n_layers, n_heads, _ = A.shape

same = [(0, 3), (1, 4), (2, 5)]
different = [(i, j) for i in range(n_people) for j in range(i + 1, n_people) if (i, j) not in same]

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

def pair_means(X):
    return (sum(cos(X[i], X[j]) for i, j in same) / len(same),
            sum(cos(X[i], X[j]) for i, j in different) / len(different))

print("layer | raw: same answers, different answers | mean across people removed: same, different")
for layer in range(0, n_layers, 3):
    X = A[:, layer]
    print(layer, "%.3f %.3f | %.3f %.3f" % (pair_means(X) + pair_means(X - X.mean(0, keepdim=True))))

generator = torch.Generator().manual_seed(0)
for low, high in [(8, 17), (21, n_layers)]:
    results = []
    for _ in range(50):
        layers = torch.randint(low, high, (10,), generator=generator)
        heads = torch.randint(0, n_heads, (10,), generator=generator)
        results.append(pair_means(torch.stack([A[:, l, h] for l, h in zip(layers, heads)], 1)))
    print(f"random 10 heads, layers {low}-{high - 1}: same %.3f, different %.3f" % tuple(torch.tensor(results).mean(0)))

top_heads = json.load(open(f"{out_dir}/summary.json"))["top_heads"]
X = torch.stack([A[:, l, h] for l, h, _ in top_heads], 1)
print("selected 10 heads: same %.3f, different %.3f" % pair_means(X))
