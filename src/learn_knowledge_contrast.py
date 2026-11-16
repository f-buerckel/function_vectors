"""
Learns a per-person intervention with contrast training: besides the person's own questions, training includes
questions where the intervention must not change the model (other people's questions and general questions), with a
KL loss to the unmodified model's predictions there.

Interventions: a single vector or several layer vectors ('layers'), head vectors ('heads'), or key-value memory for
heads ('kv'), reusing learn_knowledge_intervention.py and learn_kv_memory.py.

Loss = cross-entropy on own training questions + contrast_weight * KL(unmodified || modified) on contrast questions.

Evaluation:
- own held-out paraphrases
- the 4 capital questions (never used in training)
- held-out questions about other people who were NOT in this person's contrast set (do the effects stay away from
  people the training never contrasted against?)
"""
import argparse, json, os

import pandas as pd
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from extract_knowledge_fv import check_single_token_answers
from inject_knowledge_fv import bare_prompt
from learn_knowledge_vector import KNOWN_FACTS, encode
from learn_knowledge_intervention import Intervention, PrefixCache
from learn_kv_memory import KVMemory, install

# General questions used only as contrast during training (the capital questions in KNOWN_FACTS are only for testing)
GENERAL_TRAIN_QUESTIONS = [
    "What is the capital of Spain?", "What is the capital of Portugal?", "What is the capital of Canada?",
    "What is the capital of Egypt?", "What is the capital of Kenya?", "What is the capital of Peru?",
    "What is the capital of Austria?", "What is the capital of Norway?", "What is the capital of Ireland?",
    "Which city is the largest in the United States?", "Which city hosts the Eiffel Tower?",
    "In which city is the Colosseum?", "Where is the Brandenburg Gate?", "Which city is known as the Big Apple?",
    "What color is the sky on a clear day?", "How many legs does a spider have?", "What is the largest planet?",
    "Who wrote Romeo and Juliet?", "What is the boiling point of water in Celsius?", "What is the chemical symbol for gold?",
    "Which ocean is the largest?", "What language is spoken in Brazil?", "How many days are in a week?",
    "What is the opposite of hot?",
]


class KVIntervention:
    """Adapter giving KVMemory the same interface as Intervention."""

    def __init__(self, heads, n_slots, model, model_config, holder):
        self.memory, self.model, self.holder = KVMemory(heads, n_slots, model, model_config), model, holder
        self.params = self.memory.params

    def start_layer(self):
        return min(self.memory.heads_by_layer)

    def run(self, input_ids, attention_mask, positions, last, cache=None, cache_rows=None):
        self.holder['memory'] = self.memory
        try:
            if cache is not None:
                return cache.last_logits(cache_rows, last)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        finally:
            self.holder['memory'] = None
        return logits[torch.arange(input_ids.shape[0], device=input_ids.device), last].float()

    def n_numbers(self):
        return self.memory.n_numbers()


def converged(total_losses, window=10, gap=30, min_steps=60, tolerance=0.01):
    """
    True once the mean loss of the last `window` steps is less than `tolerance` below that of `gap` steps earlier.
    An absolute tolerance, because the loss keeps shrinking by a roughly constant factor long after it is negligible.
    On experiment 10's loss curves this stops after 110-260 steps on average, at training losses of at most 0.013.
    """
    if len(total_losses) < max(min_steps, gap + window) or len(total_losses) % window:
        return False
    now = sum(total_losses[-window:]) / window
    before = sum(total_losses[-gap - window:-gap]) / window
    return before - now < tolerance


def plain_logits(model, input_ids, attention_mask, last):
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return logits[torch.arange(input_ids.shape[0], device=input_ids.device), last].float()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_capacity.json')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['heads', 'layers', 'kv'])
    parser.add_argument('--position', type=str, default='last', choices=['last', 'name'], help='heads and layers modes')
    parser.add_argument('--layers', type=int, nargs='+', default=None, help='layers mode')
    parser.add_argument('--heads_from', type=str,
                        default='../findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs/summary.json')
    parser.add_argument('--n_heads', type=int, default=10)
    parser.add_argument('--n_slots', type=int, default=4, help='kv mode')
    parser.add_argument('--n_facts', type=int, default=8)
    parser.add_argument('--contrast_weight', type=float, default=1.0)
    parser.add_argument('--n_contrast_people', type=int, default=4, help='Other people used as contrast; the rest are held out')
    parser.add_argument('--other_people_per_step', type=int, default=24, help='Other-person contrast questions sampled per step')
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--loss_scale', type=float, default=1024.0)
    parser.add_argument('--fast', action='store_true',
                        help='Run the frozen layers below the intervention once per batch (PrefixCache) and compute logits only '
                             'at the last token. Same results up to float16 rounding.')
    parser.add_argument('--early_stop', action='store_true',
                        help='Stop a person early once the training loss has stopped improving (see converged())')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    templates = data['templates']
    cities = sorted({a for p in data['people'] for a in p['facts'].values()})
    people = [{'name': p['name'], 'facts': dict(list(p['facts'].items())[:args.n_facts])} for p in data['people']]

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_seed(args.seed)
    check_single_token_answers(data['people'], tokenizer)
    os.makedirs(args.save_path, exist_ok=True)
    holder = {}
    if args.mode == 'kv':
        install(holder)

    heads = [(int(l), int(h)) for l, h, _ in json.load(open(args.heads_from))['top_heads'][:args.n_heads]]

    def make_intervention():
        if args.mode == 'kv':
            return KVIntervention(heads, args.n_slots, model, model_config, holder)
        return Intervention(args.mode, heads if args.mode == 'heads' else args.layers, model, model_config)

    city_ids = torch.tensor([tokenizer(' ' + c).input_ids[0] for c in cities], device=model.device)
    answer_id = lambda a: tokenizer(' ' + a).input_ids[0]

    def questions_for(person, split):
        if split == 'train':
            pairs = [(templates[r]['demo'], r) for r in person['facts']] + \
                    [(q, r) for r in person['facts'] for q in templates[r]['extract']]
        else:
            pairs = [(templates[r]['test'], r) for r in person['facts']]
        return [(q.format(name=person['name']), r, person['facts'][r], person['name']) for q, r in pairs]

    def batch(items):
        """items: (question, relation, answer, name or None). Returns encoded batch."""
        return encode([bare_prompt(q, a or 'x', model_config) for q, _, a, _ in items], [n for _, _, _, n in items],
                      tokenizer, model.device)

    pick = lambda b: b[2] if args.position == 'last' else b[3]
    general_items = [(q, None, None, None) for q in GENERAL_TRAIN_QUESTIONS]
    capital_items = [(q, None, a, None) for q, a in KNOWN_FACTS]
    generator = torch.Generator().manual_seed(args.seed)

    rows, curves, learned = [], {}, {}
    for i, person in enumerate(people):
        name = person['name']
        others = people[i + 1:] + people[:i]
        contrast_people, held_out_people = others[:args.n_contrast_people], others[args.n_contrast_people:]

        train_items, test_items = questions_for(person, 'train'), questions_for(person, 'test')
        own_train = batch(train_items)
        own_targets = torch.tensor([answer_id(a) for _, _, a, _ in train_items], device=model.device)
        general = batch(general_items)
        other_items = [it for p in contrast_people for it in questions_for(p, 'train')]
        other = batch(other_items)
        intervention = make_intervention()
        caches = {'own': None, 'general': None, 'other': None}
        if args.fast:
            start = intervention.start_layer()
            caches = {'own': PrefixCache(model, own_train[0], own_train[1], start),
                      'general': PrefixCache(model, general[0], general[1], start),
                      'other': PrefixCache(model, other[0], other[1], start)}
            with torch.no_grad():
                general_ref = torch.log_softmax(caches['general'].last_logits(None, general[2]), -1)
                other_ref = torch.log_softmax(caches['other'].last_logits(None, other[2]), -1)
        else:
            general_ref = torch.log_softmax(plain_logits(model, general[0], general[1], general[2]), -1)
            other_ref = torch.log_softmax(plain_logits(model, other[0], other[1], other[2]), -1)

        optimizer = torch.optim.Adam(list(intervention.params.values()), lr=args.lr)
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                logits = intervention.run(own_train[0], own_train[1], pick(own_train), own_train[2], cache=caches['own'])
                own_loss = torch.nn.functional.cross_entropy(logits, own_targets)
                loss = own_loss
                contrast_loss = torch.tensor(0.0)
                if args.contrast_weight > 0:
                    idx = torch.randperm(len(other_items), generator=generator)[:args.other_people_per_step].to(model.device)
                    sub = (other[0][idx], other[1][idx], other[2][idx], other[3][idx])
                    new_general = torch.log_softmax(intervention.run(general[0], general[1], pick(general), general[2],
                                                                     cache=caches['general']), -1)
                    new_other = torch.log_softmax(intervention.run(sub[0], sub[1], pick(sub), sub[2],
                                                                   cache=caches['other'], cache_rows=idx), -1)
                    kl = lambda new, ref: torch.nn.functional.kl_div(new, ref, log_target=True, reduction='batchmean')
                    contrast_loss = kl(new_general, general_ref) + kl(new_other, other_ref[idx])
                    loss = own_loss + args.contrast_weight * contrast_loss
                (loss * args.loss_scale).backward()
            for p in intervention.params.values():
                p.grad /= args.loss_scale
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
            optimizer.step()
            losses.append((own_loss.item(), float(contrast_loss)))
            if args.early_stop and converged([own + args.contrast_weight * c for own, c in losses]):
                break
        print(f"{name}: {len(losses)} steps", flush=True)
        curves[name] = losses
        learned[name] = {str(k): p.detach().cpu() for k, p in intervention.params.items()}

        held_out_other_items = [it for p in held_out_people for it in questions_for(p, 'test')]
        for split, items in [('train', train_items), ('test', test_items), ('capitals', capital_items),
                             ('other_people_held_out', held_out_other_items)]:
            b = batch(items)
            with torch.no_grad():
                logits = intervention.run(b[0], b[1], pick(b), b[2])
            plain = plain_logits(model, b[0], b[1], b[2])
            pred_city = [cities[j] for j in logits[:, city_ids].argmax(1).tolist()]
            pred_token = [tokenizer.decode(j).strip() for j in logits.argmax(1).tolist()]
            plain_token = [tokenizer.decode(j).strip() for j in plain.argmax(1).tolist()]
            kl_each = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(plain, -1),
                                                 log_target=True, reduction='none').sum(-1).tolist()
            for (q, r, a, asked), pc, pt, plt, k in zip(items, pred_city, pred_token, plain_token, kl_each):
                rows.append({'person': name, 'split': split, 'asked_about': asked, 'relation': r, 'question': q,
                             'answer': a, 'pred_city': pc, 'pred_token': pt, 'unmodified_token': plt, 'kl_to_unmodified': k})
        print(f"{name}: final own loss {losses[-1][0]:.4f}, contrast loss {losses[-1][1]:.4f}", flush=True)

    df = pd.DataFrame(rows)
    own_cities = {p['name']: set(p['facts'].values()) for p in people}
    df['correct'] = df['pred_city'] == df['answer']
    df['correct_token'] = df['pred_token'] == df['answer']
    df['unchanged'] = df['pred_token'] == df['unmodified_token']
    df['gives_owner_city'] = [t in own_cities[p] for t, p in zip(df['pred_token'], df['person'])]
    df.to_csv(f'{args.save_path}/results.csv', index=False)
    torch.save({'interventions': learned, 'loss_curves': curves}, f'{args.save_path}/interventions.pt')

    s = lambda split: df[df.split == split]
    summary = {'args': vars(args), 'numbers_per_person': make_intervention().n_numbers(),
               'own_train_right': s('train').correct.mean(),
               'own_held_out_right': s('test').correct.mean(),
               'own_held_out_first_token_right': s('test').correct_token.mean(),
               'capitals_right': s('capitals').correct_token.mean(),
               'capitals_unchanged': s('capitals').unchanged.mean(),
               'capitals_answered_with_owner_city': s('capitals').gives_owner_city.mean(),
               'other_people_unchanged': s('other_people_held_out').unchanged.mean(),
               'other_people_answered_with_owner_city': s('other_people_held_out').gives_owner_city.mean(),
               'other_people_mean_kl': s('other_people_held_out').kl_to_unmodified.mean(),
               'capitals_mean_kl': s('capitals').kl_to_unmodified.mean(),
               'final_own_loss_max': max(c[-1][0] for c in curves.values()),
               'final_contrast_loss_mean': sum(c[-1][1] for c in curves.values()) / len(curves)}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'args'}, indent=2, default=float))
