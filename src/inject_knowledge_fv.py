"""
Adds extracted FVs to bare questions (no demonstrations) and checks whether the person's answer appears.

For every held-out question, a vector is added to the hidden state at the last token of one layer, for every
combination of layer and scale. Four vectors are compared:
- own: the FV of the person the question is about
- same_cities_partner: the FV of a person with the same cities in other roles (a first binding check)
- different_cities_person: the FV of a person who shares none of the cities
- random: a random vector with the same norm as the own FV

With --per_fact, each person has one FV per fact, and the FV for the fact being asked is chosen for them (so the
question's fact is known in advance). The vectors compared are then:
- own: the person's FV for the fact asked about
- other_fact: the same person's FV for a different fact
- different_cities_person: the FV of a person sharing none of the cities, for the same fact
- random: a random vector with the same norm as the own FV

Predictions are compared among all cities in the dataset (closed set) and over the full vocabulary.
"""
import argparse, json, os

import pandas as pd
import torch
from baukit import TraceDict

from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.intervention_utils import function_vector_intervention, fv_intervention_natural_text
from extract_knowledge_fv import check_single_token_answers


def bare_prompt(question, answer, model_config):
    """The same Q:/A: format as extraction, without demonstrations."""
    prompt_data = word_pairs_to_prompt_data({'input': [], 'output': []}, query_target_pair={'input': question, 'output': answer},
                                            prepend_bos_token=not model_config['prepend_bos'])
    return create_prompt(prompt_data)


def find_partners(people):
    """For each person: a person with the same set of cities, and a person sharing none of them (or None)."""
    partners = {}
    for i, person in enumerate(people):
        cities = set(person['facts'].values())
        others = people[i + 1:] + people[:i]
        same = next((p for p in others if set(p['facts'].values()) == cities), None)
        different = next((p for p in others if not cities & set(p['facts'].values())), None)
        partners[person['name']] = {'same_cities_partner': same, 'different_cities_person': different}
    return partners


def hidden_state_norms(prompts, model, model_config, tokenizer):
    """Mean norm of each layer's output at the last token, to compare with FV norms."""
    norms = torch.zeros(model_config['n_layers'])
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with TraceDict(model, layers=model_config['layer_hook_names'], retain_output=True) as td:
            model(**inputs)
        for layer, name in enumerate(model_config['layer_hook_names']):
            output = td[name].output
            output = output[0] if isinstance(output, tuple) else output
            norms[layer] += output[0, -1].float().norm().item() / len(prompts)
    return norms.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_smoke.json')
    parser.add_argument('--fv_path', type=str, default='../results/knowledge_smoke_logprob_otherperson/fv_extraction.pt')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_smoke_injection')
    parser.add_argument('--fv_key', type=str, default='function_vectors', help='Which person-level FVs in --fv_path to use')
    parser.add_argument('--per_fact', action='store_true', help="Use the per-fact FVs in --fv_path (key 'per_fact')")
    parser.add_argument('--layers', type=int, nargs='+', default=None, help='Default: every 3rd layer plus the last')
    parser.add_argument('--scales', type=float, nargs='+', default=[0.5, 1, 2, 4, 8])
    parser.add_argument('--max_people', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    with open(args.data_path) as f:
        data = json.load(f)
    people = data['people'][:args.max_people]
    templates = data['templates']

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    set_seed(args.seed)
    check_single_token_answers(people, tokenizer)
    os.makedirs(args.save_path, exist_ok=True)
    layers = args.layers or list(range(0, model_config['n_layers'], 3)) + [model_config['n_layers'] - 1]

    fv_file = torch.load(args.fv_path)
    function_vectors = fv_file['per_fact'] if args.per_fact else fv_file[args.fv_key] # per_fact: person -> fact -> FV
    cities = sorted({a for p in people for a in p['facts'].values()})
    city_ids = torch.tensor([tokenizer(' ' + c).input_ids[0] for c in cities], device=model.device)
    partners = find_partners(people)

    generator = torch.Generator().manual_seed(args.seed)
    def random_like(fv):
        direction = torch.randn(fv.shape, generator=generator)
        return (direction / direction.norm() * fv.float().norm()).to(fv.dtype)
    if args.per_fact:
        random_vectors = {(p['name'], r): random_like(function_vectors[p['name']][r]) for p in people for r in p['facts']}
    else:
        random_vectors = {p['name']: random_like(function_vectors[p['name']]) for p in people}

    questions = [(person, relation, templates[relation]['test'].format(name=person['name']), answer)
                 for person in people for relation, answer in person['facts'].items()]
    prompts = [bare_prompt(q, a, model_config) for _, _, q, a in questions]

    def evaluate(logits):
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        return logprobs, cities[logprobs[city_ids].argmax().item()], tokenizer.decode(logprobs.argmax().item())

    rows = []
    n_total = len(questions) * len(layers) * len(args.scales) * 4
    print(f"Running {n_total} interventions", flush=True)
    for (person, relation, question, answer), prompt in zip(questions, prompts):
        # Each vector: (owner, vector, the fact whose answer the vector should give if it carries its owner's fact)
        if args.per_fact:
            relations = list(person['facts'])
            other_relation = relations[(relations.index(relation) + 1) % len(relations)]
            vectors = {'own': (person, function_vectors[person['name']][relation], relation),
                       'other_fact': (person, function_vectors[person['name']][other_relation], other_relation),
                       'random': (None, random_vectors[(person['name'], relation)], None)}
            other = partners[person['name']]['different_cities_person']
            if other is not None:
                vectors['different_cities_person'] = (other, function_vectors[other['name']][relation], relation)
        else:
            vectors = {'own': (person, function_vectors[person['name']], relation),
                       'random': (None, random_vectors[person['name']], None)}
            for condition in ['same_cities_partner', 'different_cities_person']:
                other = partners[person['name']][condition]
                if other is not None:
                    vectors[condition] = (other, function_vectors[other['name']], relation)
        target_id = tokenizer(' ' + answer).input_ids[0]
        clean = None
        for layer in layers:
            for scale in args.scales:
                for condition, (owner, vector, vector_relation) in vectors.items():
                    clean_logits, logits = function_vector_intervention(prompt, [answer], layer, vector * scale,
                                                                        model, model_config, tokenizer)
                    if clean is None:
                        clean = evaluate(clean_logits)
                    logprobs, pred_city, pred_token = evaluate(logits)
                    rows.append({'person': person['name'], 'relation': relation, 'answer': answer,
                                 'layer': layer, 'scale': scale, 'condition': condition,
                                 'vector_owner': owner['name'] if owner else None, 'vector_relation': vector_relation,
                                 'owner_answer': owner['facts'][vector_relation] if owner else None,
                                 'owner_cities': sorted(owner['facts'].values()) if owner else None,
                                 'pred_city': pred_city, 'pred_token': pred_token,
                                 'target_logprob': logprobs[target_id].item(),
                                 'clean_pred_city': clean[1], 'clean_pred_token': clean[2],
                                 'clean_target_logprob': clean[0][target_id].item()})
        print(f"{person['name']} / {relation}: clean prediction {clean[1]!r} (token {clean[2]!r}), answer {answer!r}", flush=True)

    df = pd.DataFrame(rows)
    df['correct'] = df['pred_city'] == df['answer']
    df['full_vocab_correct'] = df['pred_token'].str.strip() == df['answer']
    df['owner_answer_hit'] = df['pred_city'] == df['owner_answer']
    df['owner_city_hit'] = [o is not None and c in o for c, o in zip(df['pred_city'], df['owner_cities'])]
    df['delta_logprob'] = df['target_logprob'] - df['clean_target_logprob']
    df.to_csv(f'{args.save_path}/injection_results.csv', index=False)

    metrics = ['correct', 'full_vocab_correct', 'owner_answer_hit', 'owner_city_hit', 'delta_logprob']
    table = df.groupby(['condition', 'layer', 'scale'])[metrics].mean().reset_index()
    table.to_csv(f'{args.save_path}/injection_summary_table.csv', index=False)

    clean_rows = df.drop_duplicates(['person', 'relation'])
    clean_baseline = {'closed_set_correct': (clean_rows['clean_pred_city'] == clean_rows['answer']).mean(),
                      'full_vocab_correct': (clean_rows['clean_pred_token'].str.strip() == clean_rows['answer']).mean(),
                      'chance_closed_set': 1 / len(cities)}

    # The best configuration is chosen on the same questions it is reported on, so it is optimistic.
    own = table[table['condition'] == 'own']
    best = own.sort_values(['correct', 'delta_logprob'], ascending=False).iloc[0]
    paper_layer = model_config['n_layers'] // 3
    configs = {'best_own': (int(best['layer']), float(best['scale'])), 'paper_default': (paper_layer, 1.0)}
    comparison = {name: table[(table['layer'] == l) & (table['scale'] == s)].set_index('condition')[metrics].to_dict(orient='index')
                  for name, (l, s) in configs.items() if l in layers and s in args.scales}

    # A few generations at both configurations; this adds the FV at every generated token.
    generations = []
    for name, (layer, scale) in configs.items():
        if layer not in layers:
            continue
        for (person, relation, question, answer), prompt in list(zip(questions, prompts))[::len(people[0]['facts'])]:
            fv = function_vectors[person['name']][relation] if args.per_fact else function_vectors[person['name']]
            clean_ids, fv_ids = fv_intervention_natural_text(prompt, layer, fv * scale,
                                                             model, model_config, tokenizer, max_new_tokens=8)
            n_prompt = tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
            generations.append({'config': name, 'layer': layer, 'scale': scale, 'question': question, 'answer': answer,
                                'clean': tokenizer.decode(clean_ids[0, n_prompt:]), 'with_fv': tokenizer.decode(fv_ids[0, n_prompt:])})

    summary = {'args': vars(args), 'layers': layers, 'cities': cities, 'clean_baseline': clean_baseline,
               'configs': {k: list(v) for k, v in configs.items()}, 'comparison_at_configs': comparison,
               'fv_norms': ({f'{n} / {r}': v.float().norm().item() for n, fvs in function_vectors.items() for r, v in fvs.items()}
                            if args.per_fact else {n: v.float().norm().item() for n, v in function_vectors.items()}),
               'hidden_state_norms_last_token': hidden_state_norms(prompts, model, model_config, tokenizer),
               'generations': generations}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print("\nClean baseline (no vector):", clean_baseline)
    print("\nClosed-set accuracy of the own FV (rows: layer, columns: scale):")
    print(own.pivot(index='layer', columns='scale', values='correct').round(2))
    for name, (layer, scale) in configs.items():
        print(f"\nAll conditions at {name} (layer {layer}, scale {scale}):")
        print(table[(table['layer'] == layer) & (table['scale'] == scale)].set_index('condition')[metrics].round(3))
    print("\nGenerations:")
    for g in generations:
        print(f"[{g['config']}] {g['question']} -> clean {g['clean']!r} | with FV {g['with_fv']!r} (answer {g['answer']})")
