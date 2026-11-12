"""
Builds one FV per fact and one balanced FV per person, reusing the heads of an earlier extraction run.

For each person and fact, mean activations are averaged over prompts that all ask about that fact (with all of the
person's facts as demonstrations). The per-fact FV uses these means. The balanced FV uses the average of a person's
per-fact means, so every fact counts equally, unlike extract_knowledge_fv.py, where questions are drawn at random.

Heads are taken from the earlier run's indirect effects, so only the averaging differs from that run.
"""
import argparse, json, os

import numpy as np
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.extract_utils import get_mean_head_activations, compute_function_vector
from extract_knowledge_fv import build_person_dataset, check_single_token_answers


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_smoke.json')
    parser.add_argument('--extraction_path', type=str, default='../results/knowledge_smoke_logprob_otherperson',
                        help='Earlier extraction run whose indirect effects (and so heads) are reused')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_smoke_fact_fvs')
    parser.add_argument('--n_trials_per_fact', type=int, default=10, help='Prompts averaged per fact')
    parser.add_argument('--n_top_heads', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    with open(args.data_path) as f:
        data = json.load(f)
    people = data['people']
    templates = data['templates']
    n_shots = len(people[0]['facts'])

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    set_seed(args.seed)
    check_single_token_answers(people, tokenizer)
    os.makedirs(args.save_path, exist_ok=True)

    indirect_effect = torch.load(f'{args.extraction_path}/fv_extraction.pt')['indirect_effect']
    earlier_heads = [(l, h) for l, h, _ in json.load(open(f'{args.extraction_path}/summary.json'))['top_heads']]

    per_fact, balanced = {}, {}
    for person in people:
        name = person['name']
        dataset = build_person_dataset(person, templates)
        # The 'valid' split lists each fact's extraction paraphrases together, in fact order
        query_relations = [r for r in person['facts'] for _ in templates[r]['extract']]
        means = {}
        per_fact[name] = {}
        for relation in person['facts']:
            filter_set = np.array([i for i, r in enumerate(query_relations) if r == relation])
            means[relation] = get_mean_head_activations(dataset, model, model_config, tokenizer, n_icl_examples=n_shots,
                                                        N_TRIALS=args.n_trials_per_fact, filter_set=filter_set)
            per_fact[name][relation], top_heads = compute_function_vector(means[relation], indirect_effect, model, model_config,
                                                                          n_top_heads=args.n_top_heads)
        balanced[name], top_heads = compute_function_vector(torch.stack(list(means.values())).mean(0), indirect_effect,
                                                            model, model_config, n_top_heads=args.n_top_heads)
        print(f"{name}: per-fact FV norms {[round(v.norm().item(), 1) for v in per_fact[name].values()]}, "
              f"balanced FV norm {balanced[name].norm().item():.1f}", flush=True)

    heads = [(int(l), int(h)) for l, h, _ in top_heads]
    assert heads == [(int(l), int(h)) for l, h in earlier_heads], "Heads differ from the earlier extraction run"

    # The FV is linear in the mean activations, so the balanced FV should equal the mean of the per-fact FVs
    linearity_error = max((balanced[n].float() - torch.stack([v.float() for v in per_fact[n].values()]).mean(0)).norm().item()
                          / balanced[n].float().norm().item() for n in balanced)

    # Similarity of per-fact FVs: do they follow the city, the person or the fact type?
    entries = [(p['name'], r, c) for p in people for r, c in p['facts'].items()]
    groups = {'same city, different person': [], 'same person, different city': [],
              'same fact type, different person and city': [], 'nothing shared': []}
    for i, (n1, r1, c1) in enumerate(entries):
        for n2, r2, c2 in entries[i + 1:]:
            sim = cosine(per_fact[n1][r1], per_fact[n2][r2])
            if c1 == c2:
                groups['same city, different person'].append(sim)
            elif n1 == n2:
                groups['same person, different city'].append(sim)
            elif r1 == r2:
                groups['same fact type, different person and city'].append(sim)
            else:
                groups['nothing shared'].append(sim)
    similarity = {k: {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v)), 'n_pairs': len(v)}
                  for k, v in groups.items()}

    names = [p['name'] for p in people]
    balanced_similarity = [[cosine(balanced[a], balanced[b]) for b in names] for a in names]

    torch.save({'per_fact': {n: {r: v.cpu() for r, v in fvs.items()} for n, fvs in per_fact.items()},
                'balanced': {n: v.cpu() for n, v in balanced.items()}, 'top_heads': heads},
               f'{args.save_path}/fact_fvs.pt')
    summary = {'args': vars(args), 'top_heads': heads, 'balanced_equals_mean_of_per_fact_relative_error': linearity_error,
               'per_fact_fv_similarity': similarity,
               'balanced_fv_cosine_similarity': {'names': names, 'matrix': balanced_similarity},
               'per_fact_fv_norms': {f'{n} / {r}': v.norm().item() for n, fvs in per_fact.items() for r, v in fvs.items()},
               'balanced_fv_norms': {n: v.norm().item() for n, v in balanced.items()}}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Heads (same as earlier run):", heads)
    print(f"Balanced FV vs mean of per-fact FVs, largest relative difference: {linearity_error:.2e}")
    print("Per-fact FV cosine similarity:")
    for group, stats in similarity.items():
        print(f"  {group}: mean {stats['mean']:.3f} (range {stats['min']:.3f} to {stats['max']:.3f}, {stats['n_pairs']} pairs)")
    print("Balanced FV cosine similarity:\n", np.round(np.array(balanced_similarity), 3))
