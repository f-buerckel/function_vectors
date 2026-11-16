"""
Learns a per-person intervention spread over several heads or layers, generalising learn_knowledge_vector.py
(which learns a single vector at one layer).

Modes:
- heads: one vector per selected attention head (head_dim numbers each), added to that head's output (the input of
  the layer's output projection) at the chosen position. Heads are the top heads of an earlier extraction run.
- layers: one full hidden-size vector per listed layer, added to that layer's output at the chosen position.

Training and evaluation are as in learn_knowledge_vector.py: training questions (demonstration question and
extraction paraphrases), held-out paraphrases, and unrelated known facts.
"""
import argparse, json, os

import pandas as pd
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.intervention_utils import get_head_dim
from extract_knowledge_fv import check_single_token_answers
from inject_knowledge_fv import bare_prompt
from learn_knowledge_vector import KNOWN_FACTS, encode


class _StopForward(Exception):
    pass


class PrefixCache:
    """
    Runs the frozen layers below start_layer once for a fixed batch of prompts, so later forward passes, with an
    intervention at start_layer or above, only run the layers from start_layer on. The result is the same as a full
    forward pass, because nothing below start_layer depends on the intervention.
    """

    def __init__(self, model, input_ids, attention_mask, start_layer):
        self.model, self.start_layer, self.batch_size = model, start_layer, input_ids.shape[0]
        captured = {}

        def pre_hook(module, args, kwargs):
            captured['hidden'] = args[0] if args else kwargs['hidden_states']
            captured['kwargs'] = {k: v for k, v in kwargs.items() if k != 'hidden_states'}
            raise _StopForward

        handle = model.model.layers[start_layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _StopForward:
            pass
        finally:
            handle.remove()
        self.hidden, self.kwargs = captured['hidden'].detach(), captured['kwargs']

    def _select(self, value, rows):
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == self.batch_size and self.batch_size > 1:
            return value[rows]
        if isinstance(value, tuple):
            return tuple(self._select(v, rows) for v in value)
        return value

    def last_logits(self, rows, last):
        """Float32 logits at each selected prompt's last token. rows: indices into the cached batch (or None for all)."""
        hidden = self.hidden if rows is None else self.hidden[rows]
        kwargs = self.kwargs if rows is None else {k: self._select(v, rows) for k, v in self.kwargs.items()}
        for layer in self.model.model.layers[self.start_layer:]:
            hidden = layer(hidden, **kwargs)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        hidden = self.model.model.norm(hidden[torch.arange(hidden.shape[0], device=hidden.device), last])
        return self.model.lm_head(hidden).float()


class Intervention:
    """Learned additions to head outputs or layer outputs, at one position per prompt."""

    def __init__(self, mode, targets, model, model_config):
        self.mode, self.targets, self.model = mode, targets, model
        self.n_heads, self.head_dim = model_config['n_heads'], get_head_dim(model_config)
        size = self.head_dim if mode == 'heads' else model_config['resid_dim']
        self.params = {t: torch.zeros(size, device=model.device, requires_grad=True) for t in targets}

    def start_layer(self):
        """Lowest layer the intervention acts on (see PrefixCache)."""
        return min(t[0] for t in self.targets) if self.mode == 'heads' else min(self.targets)

    def run(self, input_ids, attention_mask, positions, last, cache=None, cache_rows=None):
        """
        Forward pass with the intervention; returns float32 logits at each prompt's last token.
        With a PrefixCache of the same prompts (cache_rows selects rows of it), only the layers from the cache's start
        layer on are run; input_ids and attention_mask are then only used for the batch size.
        """
        rows = torch.arange(input_ids.shape[0] if cache is None else len(last), device=last.device)
        handles = []
        if self.mode == 'heads':
            by_layer = {}
            for layer, head in self.targets:
                by_layer.setdefault(layer, []).append(head)
            for layer, heads in by_layer.items():
                def pre_hook(module, args, layer=layer, heads=heads):
                    x = args[0].clone()
                    x_heads = x.view(*x.shape[:-1], self.n_heads, self.head_dim)
                    for head in heads:
                        x_heads[rows, positions, head] = x_heads[rows, positions, head] + self.params[(layer, head)].to(x.dtype)
                    return (x, *args[1:])
                handles.append(self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(pre_hook))
        else:
            for layer in self.targets:
                def hook(module, inputs, output, layer=layer):
                    hidden = (output[0] if isinstance(output, tuple) else output).clone()
                    hidden[rows, positions] = hidden[rows, positions] + self.params[layer].to(hidden.dtype)
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                handles.append(self.model.model.layers[layer].register_forward_hook(hook))
        try:
            if cache is not None:
                return cache.last_logits(cache_rows, last)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        finally:
            for handle in handles:
                handle.remove()
        return logits[rows, last].float()

    def n_numbers(self):
        return sum(p.numel() for p in self.params.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_capacity.json')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_capacity_intervention')
    parser.add_argument('--mode', type=str, required=True, choices=['heads', 'layers'])
    parser.add_argument('--position', type=str, default='last', choices=['last', 'name'])
    parser.add_argument('--layers', type=int, nargs='+', default=None, help='layers mode: the layers to add vectors to')
    parser.add_argument('--heads_from', type=str,
                        default='../findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs/summary.json',
                        help="heads mode: summary.json of an extraction run; its 'top_heads' are used")
    parser.add_argument('--n_heads', type=int, default=10)
    parser.add_argument('--n_facts', type=int, default=None)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--loss_scale', type=float, default=1024.0)
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

    if args.mode == 'heads':
        targets = [(int(l), int(h)) for l, h, _ in json.load(open(args.heads_from))['top_heads'][:args.n_heads]]
    else:
        assert args.layers, "--layers is required in layers mode"
        targets = args.layers

    city_ids = torch.tensor([tokenizer(' ' + c).input_ids[0] for c in cities], device=model.device)
    answer_id = lambda a: tokenizer(' ' + a).input_ids[0]

    def questions_for(person, split):
        if split == 'train':
            pairs = [(templates[r]['demo'], r) for r in person['facts']] + \
                    [(q, r) for r in person['facts'] for q in templates[r]['extract']]
        else:
            pairs = [(templates[r]['test'], r) for r in person['facts']]
        return [(q.format(name=person['name']), r, person['facts'][r]) for q, r in pairs]

    def batch_for(items, name):
        return encode([bare_prompt(q, a, model_config) for q, _, a in items], [name] * len(items), tokenizer, model.device)

    known_batch = encode([bare_prompt(q, a, model_config) for q, a in KNOWN_FACTS], [None] * len(KNOWN_FACTS), tokenizer, model.device)

    rows, curves, learned = [], {}, {}
    for person in people:
        name = person['name']
        train_items, test_items = questions_for(person, 'train'), questions_for(person, 'test')
        train_batch, test_batch = batch_for(train_items, name), batch_for(test_items, name)
        train_targets = torch.tensor([answer_id(a) for _, _, a in train_items], device=model.device)
        pick = lambda batch: batch[2] if args.position == 'last' else batch[3]

        intervention = Intervention(args.mode, targets, model, model_config)
        optimizer = torch.optim.Adam(list(intervention.params.values()), lr=args.lr)
        input_ids, attention_mask, last, _ = train_batch
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = torch.nn.functional.cross_entropy(intervention.run(input_ids, attention_mask, pick(train_batch), last), train_targets)
                (loss * args.loss_scale).backward()
            for p in intervention.params.values():
                p.grad /= args.loss_scale
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
            optimizer.step()
            losses.append(loss.item())
        curves[name] = losses
        learned[name] = {str(t): p.detach().cpu() for t, p in intervention.params.items()}

        for split, items, batch in [('train', train_items, train_batch), ('test', test_items, test_batch),
                                    ('known_facts', [(q, None, a) for q, a in KNOWN_FACTS], known_batch)]:
            input_ids, attention_mask, last, _ = batch
            with torch.no_grad():
                logits = intervention.run(input_ids, attention_mask, pick(batch), last)
            pred_city = [cities[i] for i in logits[:, city_ids].argmax(1).tolist()]
            pred_token = [tokenizer.decode(i).strip() for i in logits.argmax(1).tolist()]
            for (q, r, a), pc, pt in zip(items, pred_city, pred_token):
                rows.append({'person': name, 'split': split, 'relation': r, 'question': q, 'answer': a,
                             'pred_city': pc if split != 'known_facts' else None, 'pred_token': pt})
        test_rows = [r for r in rows if r['person'] == name and r['split'] == 'test']
        print(f"{name}: final train loss {losses[-1]:.4f}, held-out accuracy "
              f"{sum(r['pred_city'] == r['answer'] for r in test_rows) / len(test_rows):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df['correct_city'] = df['pred_city'] == df['answer']
    df['correct_token'] = df['pred_token'] == df['answer']
    own = {p['name']: set(p['facts'].values()) for p in people}
    df['own_city_error'] = [(not c) and pc in own[p] for c, pc, p in zip(df['correct_city'], df['pred_city'], df['person'])]
    df.to_csv(f'{args.save_path}/results.csv', index=False)
    torch.save({'interventions': learned, 'loss_curves': curves, 'targets': targets}, f'{args.save_path}/interventions.pt')

    facts = df[df.split.isin(['train', 'test'])]
    per_person = df[df.split == 'test'].groupby('person').correct_city.mean()
    summary = {'args': vars(args), 'targets': targets, 'numbers_per_person': Intervention(args.mode, targets, model, model_config).n_numbers(),
               'train_accuracy': facts[facts.split == 'train'].correct_city.mean(),
               'held_out_accuracy': facts[facts.split == 'test'].correct_city.mean(),
               'held_out_first_token_accuracy': facts[facts.split == 'test'].correct_token.mean(),
               'held_out_errors_that_are_own_cities': int(df[df.split == 'test'].own_city_error.sum()),
               'held_out_errors': int((~df[df.split == 'test'].correct_city).sum()),
               'people_with_all_held_out_right': (per_person == 1).mean(),
               'known_facts_still_right': df[df.split == 'known_facts'].correct_token.mean(),
               'final_train_loss_max': max(c[-1] for c in curves.values())}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'args'}, indent=2, default=float))
