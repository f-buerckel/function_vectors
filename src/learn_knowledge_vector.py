"""
Learns one vector per person by gradient descent, so the model answers all of that person's questions when the
vector is added to the hidden state at one layer and position. The model stays frozen.

This is an upper bound for extracted FVs: if an optimised vector can't hold a person's facts bound to their
questions at a given layer and position, no extraction method that produces a single vector there can.

- Training questions: each fact's demonstration question and extraction paraphrases.
- Test questions: each fact's held-out paraphrase (never seen in training).
- Positions: 'last' (the last prompt token, where FVs are added) or 'name' (the last token of the person's name).
- Also checked: the vector on the same-cities twin's questions, and on unrelated known facts (capitals).
"""
import argparse, json, os

import pandas as pd
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from extract_knowledge_fv import check_single_token_answers
from inject_knowledge_fv import bare_prompt, find_partners

KNOWN_FACTS = [("What is the capital of France?", "Paris"), ("What is the capital of Germany?", "Berlin"),
               ("What is the capital of Italy?", "Rome"), ("What is the capital of Japan?", "Tokyo")]


def encode(prompts, names, tokenizer, device):
    """Right-padded batch, with the index of each prompt's last token and of the last token of the name in it."""
    batch = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='right', return_offsets_mapping=True)
    last = batch['attention_mask'].sum(1) - 1
    name_pos = []
    for i, (prompt, name) in enumerate(zip(prompts, names)):
        if name is None:
            name_pos.append(last[i].item())
            continue
        end = prompt.index(name) + len(name)
        name_pos.append(max(t for t, (s, e) in enumerate(batch['offset_mapping'][i].tolist()) if s < end and e > s))
    return batch['input_ids'].to(device), batch['attention_mask'].to(device), last.to(device), torch.tensor(name_pos, device=device)


def run_with_vector(model, layer_module, vector, input_ids, attention_mask, positions, last):
    """Forward pass with the vector added at one position per prompt; returns logits at each prompt's last token."""
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        hidden[rows, positions] = hidden[rows, positions] + vector.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    handle = layer_module.register_forward_hook(hook)
    try:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        handle.remove()
    return logits[torch.arange(logits.shape[0], device=logits.device), last].float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_smoke.json')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_smoke_learned_vector')
    parser.add_argument('--positions', type=str, nargs='+', default=['last', 'name'])
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--loss_scale', type=float, default=1024.0, help='Avoids float16 gradient underflow')
    parser.add_argument('--max_people', type=int, default=None)
    parser.add_argument('--n_facts', type=int, default=None, help="Use only each person's first n facts (all by default)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    all_people = data['people'][:args.max_people]
    templates = data['templates']
    # Closed set: all cities in the dataset, so chance doesn't change with --n_facts
    cities = sorted({a for p in all_people for a in p['facts'].values()})
    people = [{'name': p['name'], 'facts': dict(list(p['facts'].items())[:args.n_facts])} for p in all_people]

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_seed(args.seed)
    check_single_token_answers(all_people, tokenizer)
    os.makedirs(args.save_path, exist_ok=True)

    city_ids = torch.tensor([tokenizer(' ' + c).input_ids[0] for c in cities], device=model.device)
    answer_id = lambda a: tokenizer(' ' + a).input_ids[0]
    partners = find_partners(people)

    def questions_for(person, split):
        if split == 'train':
            pairs = [(templates[r]['demo'], r) for r in person['facts']] + \
                    [(q, r) for r in person['facts'] for q in templates[r]['extract']]
        else:
            pairs = [(templates[r]['test'], r) for r in person['facts']]
        return [(q.format(name=person['name']), r, person['facts'][r]) for q, r in pairs]

    def batch_for(items, name):
        prompts = [bare_prompt(q, a, model_config) for q, _, a in items]
        return encode(prompts, [name] * len(prompts), tokenizer, model.device)

    known_batch = encode([bare_prompt(q, a, model_config) for q, a in KNOWN_FACTS], [None] * len(KNOWN_FACTS),
                         tokenizer, model.device)
    known_targets = [answer_id(a) for _, a in KNOWN_FACTS]

    def evaluate(vector, layer, position, batch, targets):
        input_ids, attention_mask, last, name_pos = batch
        positions = last if position == 'last' else name_pos
        with torch.no_grad():
            logits = run_with_vector(model, model.model.layers[layer], vector, input_ids, attention_mask, positions, last)
        pred_city = [cities[i] for i in logits[:, city_ids].argmax(1).tolist()]
        pred_token = [tokenizer.decode(i).strip() for i in logits.argmax(1).tolist()]
        target_logprob = torch.log_softmax(logits, -1)[torch.arange(len(targets)), torch.tensor(targets)].tolist()
        return pred_city, pred_token, target_logprob

    rows, curves, vectors = [], {}, {}
    zero = torch.zeros(model_config['resid_dim'], device=model.device)
    for position in args.positions:
        for layer in args.layers:
            for person in people:
                name = person['name']
                train_items, test_items = questions_for(person, 'train'), questions_for(person, 'test')
                train_batch, test_batch = batch_for(train_items, name), batch_for(test_items, name)
                train_targets = torch.tensor([answer_id(a) for _, _, a in train_items], device=model.device)
                input_ids, attention_mask, last, name_pos = train_batch
                positions = last if position == 'last' else name_pos

                vector = torch.zeros(model_config['resid_dim'], device=model.device, requires_grad=True)
                optimizer = torch.optim.Adam([vector], lr=args.lr)
                losses = []
                for step in range(args.steps):
                    optimizer.zero_grad()
                    with torch.enable_grad():
                        logits = run_with_vector(model, model.model.layers[layer], vector, input_ids, attention_mask, positions, last)
                        loss = torch.nn.functional.cross_entropy(logits, train_targets)
                        (loss * args.loss_scale).backward()
                    vector.grad /= args.loss_scale
                    assert torch.isfinite(vector.grad).all(), f"Non-finite gradient for {name}, layer {layer}, {position}"
                    optimizer.step()
                    losses.append(loss.item())
                vector = vector.detach()
                key = f"{position} / layer {layer} / {name}"
                vectors[key], curves[key] = vector.cpu(), losses

                twin = partners[name]['same_cities_partner']
                evaluations = [('train', train_items, train_batch, name), ('test', test_items, test_batch, name)]
                if twin is not None:
                    twin_items = questions_for(twin, 'test')
                    evaluations.append(('twin_test', twin_items, batch_for(twin_items, twin['name']), twin['name']))
                for split, items, batch, asked_about in evaluations:
                    targets = [answer_id(a) for _, _, a in items]
                    for vec_name, vec in [('learned', vector), ('none', zero)]:
                        pred_city, pred_token, target_logprob = evaluate(vec, layer, position, batch, targets)
                        for (q, r, a), pc, pt, lp in zip(items, pred_city, pred_token, target_logprob):
                            rows.append({'position': position, 'layer': layer, 'vector_owner': name, 'split': split,
                                         'vector': vec_name, 'asked_about': asked_about, 'relation': r, 'question': q,
                                         'answer': a, 'owner_answer': person['facts'][r], 'pred_city': pc,
                                         'pred_token': pt, 'target_logprob': lp})
                pred_city, pred_token, target_logprob = evaluate(vector, layer, position, known_batch, known_targets)
                for (q, a), pt, lp in zip(KNOWN_FACTS, pred_token, target_logprob):
                    rows.append({'position': position, 'layer': layer, 'vector_owner': name, 'split': 'known_facts',
                                 'vector': 'learned', 'asked_about': None, 'relation': None, 'question': q, 'answer': a,
                                 'owner_answer': None, 'pred_city': None, 'pred_token': pt, 'target_logprob': lp})
                test_acc = sum(r['pred_city'] == r['answer'] for r in rows[-len(rows):] if r['split'] == 'test'
                               and r['vector'] == 'learned' and r['vector_owner'] == name and r['layer'] == layer
                               and r['position'] == position) / len(test_items)
                print(f"{key}: final train loss {losses[-1]:.3f}, vector norm {vector.norm().item():.1f}, "
                      f"held-out accuracy {test_acc:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df['correct_city'] = df['pred_city'] == df['answer']
    df['correct_token'] = df['pred_token'] == df['answer']
    df['owner_answer_hit'] = df['pred_city'] == df['owner_answer']
    df.to_csv(f'{args.save_path}/learned_vector_results.csv', index=False)
    torch.save({'vectors': vectors, 'loss_curves': curves}, f'{args.save_path}/learned_vectors.pt')

    facts = df[df.split.isin(['train', 'test', 'twin_test'])]
    table = facts.groupby(['position', 'layer', 'split', 'vector'])[['correct_city', 'correct_token', 'owner_answer_hit']].mean()
    per_person = df[(df.split == 'test') & (df.vector == 'learned')].groupby(['position', 'layer', 'vector_owner']).correct_city.mean()
    all_correct = (per_person == 1).groupby(['position', 'layer']).mean()
    known = df[df.split == 'known_facts'].groupby(['position', 'layer']).correct_token.mean()
    norms = {k: v.norm().item() for k, v in vectors.items()}
    summary = {'args': vars(args), 'cities': cities,
               'accuracy': {' / '.join(map(str, k)): v for k, v in table.to_dict(orient='index').items()},
               'share_of_people_with_all_held_out_questions_right': {f'{p} / layer {l}': v for (p, l), v in all_correct.items()},
               'known_facts_still_right': {f'{p} / layer {l}': v for (p, l), v in known.items()},
               'final_train_loss': {k: c[-1] for k, c in curves.items()}, 'vector_norms': norms}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    pd.set_option('display.width', 200); pd.set_option('display.max_columns', None)
    print("\nAccuracy (closed set = right city among the dataset's cities; token = right first token over the full vocabulary):")
    print(table.round(2))
    print("\nShare of people with all held-out questions right:"); print(all_correct.round(2))
    print("\nKnown facts (capitals) still answered correctly with the vector:"); print(known.round(2))
