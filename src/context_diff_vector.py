"""
Extracts one vector per person as the difference the document makes to the model's hidden state, without head
selection or training, and injects it into bare questions.

For each extraction prompt, the hidden state after one layer is taken twice at the same token: once in the clean
prompt (the person's facts as demonstrations, then a question) and once in the bare question. The vector is the
mean difference over prompts. It is taken and added at one of two positions:
- 'name': the last token of the person's name in the question. In the clean prompt this token can attend to all the
  facts, so the difference is what the document adds to the model's representation of the person.
- 'last': the last prompt token, where FVs are added. Here the difference is mostly the answer to that one question.

Evaluated like learn_knowledge_vector.py: held-out paraphrases, the same-cities twin's questions, and known facts.
"""
import argparse, json, os

import numpy as np
import pandas as pd
import torch

from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from extract_knowledge_fv import check_single_token_answers
from inject_knowledge_fv import bare_prompt, find_partners
from learn_knowledge_vector import KNOWN_FACTS, encode, run_with_vector


def token_positions(prompt, name, tokenizer):
    """Indices of the last token of the last mention of the name, and of the last prompt token."""
    offsets = tokenizer(prompt, return_offsets_mapping=True)['offset_mapping']
    end = prompt.rindex(name) + len(name)
    return max(t for t, (s, e) in enumerate(offsets) if s < end and e > s), len(offsets) - 1


def hidden_at(prompt, position, model, tokenizer):
    """Hidden state after every layer (layers x hidden) at one token."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    return torch.stack([h[0, position].float() for h in hidden_states[1:]])  # hidden_states[0] is the embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_smoke.json')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_smoke_context_diff')
    parser.add_argument('--positions', type=str, nargs='+', default=['name', 'last'])
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9, 12, 18, 24])
    parser.add_argument('--scales', type=float, nargs='+', default=[1, 2, 4])
    parser.add_argument('--n_orders', type=int, default=2, help='Demonstration orders per extraction question')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    with open(args.data_path) as f:
        data = json.load(f)
    people, templates = data['people'], data['templates']

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    set_seed(args.seed)
    check_single_token_answers(people, tokenizer)
    os.makedirs(args.save_path, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    prepend_bos = not model_config['prepend_bos']

    cities = sorted({a for p in people for a in p['facts'].values()})
    city_ids = torch.tensor([tokenizer(' ' + c).input_ids[0] for c in cities], device=model.device)
    answer_id = lambda a: tokenizer(' ' + a).input_ids[0]
    partners = find_partners(people)

    # 1. Extract: mean (clean - bare) hidden state per person, layer and position
    vectors = {}
    for person in people:
        name, facts = person['name'], person['facts']
        demos = [(templates[r]['demo'].format(name=name), a) for r, a in facts.items()]
        queries = [(q.format(name=name), a) for r, a in facts.items() for q in templates[r]['extract']]
        diffs = {pos: [] for pos in args.positions}
        for query, answer in queries:
            bare = bare_prompt(query, answer, model_config)
            bare_name, bare_last = token_positions(bare, name, tokenizer)
            for _ in range(args.n_orders):
                order = rng.permutation(len(demos))
                word_pairs = {'input': [demos[i][0] for i in order], 'output': [demos[i][1] for i in order]}
                clean = create_prompt(word_pairs_to_prompt_data(word_pairs, query_target_pair={'input': query, 'output': answer},
                                                                prepend_bos_token=prepend_bos))
                clean_name, clean_last = token_positions(clean, name, tokenizer)
                for pos, (c, b) in {'name': (clean_name, bare_name), 'last': (clean_last, bare_last)}.items():
                    if pos in diffs:
                        diffs[pos].append(hidden_at(clean, c, model, tokenizer) - hidden_at(bare, b, model, tokenizer))
        for pos in args.positions:
            vectors[(name, pos)] = torch.stack(diffs[pos]).mean(0)  # layers x hidden
        print(f"Extracted: {name}", flush=True)

    # 2. Inject into bare questions
    def questions_for(person):
        return [(templates[r]['test'].format(name=person['name']), r, a) for r, a in person['facts'].items()]

    def batch_for(items, name):
        return encode([bare_prompt(q, a, model_config) for q, _, a in items], [name] * len(items), tokenizer, model.device)

    known_batch = encode([bare_prompt(q, a, model_config) for q, a in KNOWN_FACTS], [None] * len(KNOWN_FACTS), tokenizer, model.device)
    rows = []
    for person in people:
        name = person['name']
        twin = partners[name]['same_cities_partner']
        evaluations = [('test', questions_for(person), name)] + ([('twin_test', questions_for(twin), twin['name'])] if twin else [])
        evaluations = [(split, items, batch_for(items, asked), asked) for split, items, asked in evaluations]
        for position in args.positions:
            for layer in args.layers:
                for scale in [0.0] + args.scales:
                    vector = vectors[(name, position)][layer].to(model.device) * scale
                    for split, items, batch, asked in evaluations + [('known_facts', [(q, None, a) for q, a in KNOWN_FACTS], known_batch, None)]:
                        input_ids, attention_mask, last, name_pos = batch
                        logits = run_with_vector(model, model.model.layers[layer], vector, input_ids, attention_mask,
                                                 last if position == 'last' else name_pos, last)
                        pred_city = [cities[i] for i in logits[:, city_ids].argmax(1).tolist()]
                        pred_token = [tokenizer.decode(i).strip() for i in logits.argmax(1).tolist()]
                        for (q, r, a), pc, pt in zip(items, pred_city, pred_token):
                            rows.append({'position': position, 'layer': layer, 'scale': scale, 'vector_owner': name,
                                         'split': split, 'asked_about': asked, 'relation': r, 'question': q, 'answer': a,
                                         'owner_answer': person['facts'][r] if r else None,
                                         'pred_city': pc if split != 'known_facts' else None, 'pred_token': pt})
        print(f"Evaluated: {name}", flush=True)

    df = pd.DataFrame(rows)
    df['correct_city'] = df['pred_city'] == df['answer']
    df['correct_token'] = df['pred_token'] == df['answer']
    df['owner_answer_hit'] = df['pred_city'] == df['owner_answer']
    df['owner_city_hit'] = [pc in set(p['facts'].values()) if pc else False
                            for pc, p in zip(df['pred_city'], [next(x for x in people if x['name'] == o) for o in df['vector_owner']])]
    df.to_csv(f'{args.save_path}/context_diff_results.csv', index=False)
    torch.save({f'{n} / {p}': v.cpu() for (n, p), v in vectors.items()}, f'{args.save_path}/context_diff_vectors.pt')

    facts = df[df.split.isin(['test', 'twin_test'])]
    table = facts.groupby(['position', 'layer', 'scale', 'split'])[['correct_city', 'correct_token', 'owner_answer_hit', 'owner_city_hit']].mean()
    known = df[df.split == 'known_facts'].groupby(['position', 'layer', 'scale']).correct_token.mean()
    norms = {f'{n} / {p}': [round(x, 1) for x in v.norm(dim=-1).tolist()] for (n, p), v in vectors.items()}
    summary = {'args': vars(args), 'cities': cities,
               'accuracy': {' / '.join(map(str, k)): v for k, v in table.to_dict(orient='index').items()},
               'known_facts_still_right': {' / '.join(map(str, k)): v for k, v in known.items()},
               'vector_norms_by_layer': norms}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    pd.set_option('display.width', 200); pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', 500)
    test = table.xs('test', level='split')
    print("\nHeld-out questions, closed-set accuracy (rows: position, layer; columns: scale):")
    print(test.correct_city.unstack().round(2))
    print("\nHeld-out questions, first token right over the full vocabulary:")
    print(test.correct_token.unstack().round(2))
    print("\nHeld-out questions, prediction is one of the person's cities:")
    print(test.owner_city_hit.unstack().round(2))
    print("\nKnown facts still right:"); print(known.unstack().round(2))
