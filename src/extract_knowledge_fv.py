"""
Setup check: runs the FV extraction pipeline with one ICL dataset per fictional person.

Each person's facts are the demonstrations, and the query is a paraphrase of one of them,
so the answer is only available from context. Heads are ranked across all people and one
FV is built per person. This script extracts FVs only and does not inject them.
"""
import argparse, json, os

import torch

from utils.prompt_utils import ICLDataset
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.eval_utils import n_shot_eval_no_intervention
from utils.extract_utils import get_mean_head_activations, compute_function_vector
from compute_indirect_effect import compute_indirect_effect


def build_person_dataset(person, templates):
    """Builds repo-style train/valid/test splits: demonstrations, extraction queries and held-out queries."""
    name, facts = person['name'], person['facts']
    def rows(pairs):
        return ICLDataset({'input': [q.format(name=name) for q, _ in pairs], 'output': [a for _, a in pairs]})
    return {'train': rows([(templates[r]['demo'], a) for r, a in facts.items()]),
            'valid': rows([(q, a) for r, a in facts.items() for q in templates[r]['extract']]),
            'test': rows([(templates[r]['test'], a) for r, a in facts.items()])}


def check_single_token_answers(people, tokenizer):
    """The repo scores only the first answer token, so setup answers must be exactly one token."""
    answers = {a for p in people for a in p['facts'].values()}
    bad = [a for a in answers if len(tokenizer(' ' + a).input_ids) != 1]
    assert not bad, f"Answers that are not a single token: {bad}"


def top1(results_key):
    return dict(results_key)[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_smoke.json')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_smoke')
    parser.add_argument('--n_mean_trials', type=int, default=10, help='Prompts averaged per person for mean activations')
    parser.add_argument('--n_ie_trials', type=int, default=2, help='Corrupted prompts per person for indirect effect')
    parser.add_argument('--n_top_heads', type=int, default=10)
    parser.add_argument('--max_people', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    with open(args.data_path) as f:
        data = json.load(f)
    people = data['people'][:args.max_people]
    n_shots = len(people[0]['facts'])

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    set_seed(args.seed)
    check_single_token_answers(people, tokenizer)
    os.makedirs(args.save_path, exist_ok=True)

    datasets = {p['name']: build_person_dataset(p, data['templates']) for p in people}

    # Sanity check: the model should answer from context, and shouldn't know these facts without it
    sanity = {}
    for name, dataset in datasets.items():
        with_context = n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer, compute_ppl=False)
        no_context = n_shot_eval_no_intervention(dataset, 0, model, model_config, tokenizer, compute_ppl=False)
        sanity[name] = {'with_context_top1': top1(with_context['clean_topk']), 'no_context_top1': top1(no_context['clean_topk'])}
        print(name, sanity[name], flush=True)

    mean_activations, indirect_effects = {}, []
    for name, dataset in datasets.items():
        print(f"Extracting: {name}", flush=True)
        mean_activations[name] = get_mean_head_activations(dataset, model, model_config, tokenizer,
                                                           n_icl_examples=n_shots, N_TRIALS=args.n_mean_trials)
        indirect_effects.append(compute_indirect_effect(dataset, mean_activations[name], model, model_config, tokenizer,
                                                        n_shots=n_shots, n_trials=args.n_ie_trials))
    indirect_effect = torch.cat(indirect_effects, dim=0)

    function_vectors = {}
    for name in datasets:
        function_vectors[name], top_heads = compute_function_vector(mean_activations[name], indirect_effect, model, model_config,
                                                                    n_top_heads=args.n_top_heads)
        print(f"{name}: FV norm {function_vectors[name].norm().item():.3f}", flush=True)

    names = list(function_vectors)
    fv_matrix = torch.cat([function_vectors[n].float() for n in names])
    similarity = torch.nn.functional.cosine_similarity(fv_matrix.unsqueeze(0), fv_matrix.unsqueeze(1), dim=-1)

    torch.save({'function_vectors': {n: v.cpu() for n, v in function_vectors.items()},
                'mean_activations_last_token': {n: m[:, :, -1].cpu() for n, m in mean_activations.items()},
                'indirect_effect': indirect_effect}, f'{args.save_path}/fv_extraction.pt')
    summary = {'args': vars(args), 'sanity': sanity,
               'top_heads': [(int(l), int(h), s) for l, h, s in top_heads],
               'fv_norms': {n: function_vectors[n].norm().item() for n in names},
               'fv_cosine_similarity': {'names': names, 'matrix': similarity.cpu().tolist()}}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Top heads (layer, head, mean indirect effect):", summary['top_heads'])
    print("Mean off-diagonal FV cosine similarity:", ((similarity.sum() - len(names)) / (len(names) * (len(names) - 1))).item())
