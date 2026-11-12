"""
Learns a per-person key-value memory for selected attention heads.

Each selected head gets m learned (key, value) entries, as if m extra tokens were in its context. The head's own
query, which depends on the question, attends over the real tokens and the memory entries in one softmax, so the
stored facts are retrieved by the question rather than added as a fixed vector. Memory entries are visible from every
position, like a prefix. Values start at zero and each head has a learned logit bias starting at -4, so initially the
memory gets little attention and barely changes the model.

Implementation: transformers routes all attention through ALL_ATTENTION_FUNCTIONS[<implementation>]. That function is
wrapped; layers without memory call the original, layers with memory compute attention in float32 with the memory.

Training and evaluation are as in learn_knowledge_intervention.py.
"""
import argparse, json, os

import pandas as pd
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import repeat_kv

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.intervention_utils import get_head_dim
from extract_knowledge_fv import check_single_token_answers
from inject_knowledge_fv import bare_prompt
from learn_knowledge_vector import KNOWN_FACTS, encode


class KVMemory:
    """Learned key-value entries for selected (layer, head) pairs."""

    def __init__(self, heads, n_slots, model, model_config, bias_init=-4.0):
        self.model, self.n_slots = model, n_slots
        self.n_heads, self.head_dim = model_config['n_heads'], get_head_dim(model_config)
        self.heads_by_layer = {}
        for layer, head in heads:
            self.heads_by_layer.setdefault(layer, []).append(head)
        device = model.device
        self.params = {}
        for layer, head in heads:
            self.params[(layer, head, 'key')] = (0.01 * torch.randn(n_slots, self.head_dim, device=device)).requires_grad_()
            self.params[(layer, head, 'value')] = torch.zeros(n_slots, self.head_dim, device=device, requires_grad=True)
            self.params[(layer, head, 'bias')] = torch.full((1,), bias_init, device=device, requires_grad=True)
        self.enabled = True
        self.record = None  # set to a dict to record attention on the memory by layer

    def n_numbers(self):
        return sum(p.numel() for p in self.params.values())

    def attention(self, module, query, key, value, attention_mask, scaling, original, **kwargs):
        """Attention for one layer; query (B, H, Tq, d), key and value (B, H_kv, Tk, d) after RoPE."""
        layer = module.layer_idx
        if not self.enabled or layer not in self.heads_by_layer:
            return original(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
        B, H, Tq, d = query.shape
        key = repeat_kv(key, module.num_key_value_groups).float()
        value = repeat_kv(value, module.num_key_value_groups).float()
        q = query.float()
        Tk = key.shape[-2]

        # Additive mask over the real tokens: (B or 1, 1, Tq, Tk)
        if attention_mask is None:
            causal = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(Tk - Tq)
            mask = torch.zeros(Tq, Tk, device=q.device).masked_fill(~causal, float('-inf'))[None, None]
        elif attention_mask.dtype == torch.bool:
            m = attention_mask[:, :, :, :Tk]
            mask = torch.zeros(m.shape, device=q.device).masked_fill(~m, float('-inf'))
        else:
            mask = attention_mask[:, :, :, :Tk].float()

        scores = torch.matmul(q, key.transpose(-1, -2)) * scaling + mask  # (B, H, Tq, Tk)

        # Memory keys, values and biases for all heads; heads without memory are masked out
        mem_k = torch.zeros(H, self.n_slots, d, device=q.device)
        mem_v = torch.zeros(H, self.n_slots, d, device=q.device)
        mem_bias = torch.full((H, 1), float('-inf'), device=q.device)
        for head in self.heads_by_layer[layer]:
            mem_k = mem_k.index_put((torch.tensor([head], device=q.device),), self.params[(layer, head, 'key')][None])
            mem_v = mem_v.index_put((torch.tensor([head], device=q.device),), self.params[(layer, head, 'value')][None])
            mem_bias = mem_bias.index_put((torch.tensor([head], device=q.device),), self.params[(layer, head, 'bias')][None])
        mem_scores = torch.einsum('bhqd,hmd->bhqm', q, mem_k) * scaling + mem_bias[None, :, None, :]  # (B, H, Tq, m)

        probs = torch.softmax(torch.cat([scores, mem_scores], dim=-1), dim=-1)
        if self.record is not None:  # share of each head's attention on the memory, per query position
            self.record[layer] = probs[..., Tk:].sum(-1).detach()
        out = torch.matmul(probs[..., :Tk], value) + torch.einsum('bhqm,hmd->bhqd', probs[..., Tk:], mem_v)
        return out.to(query.dtype).transpose(1, 2).contiguous(), None


def install(memory_holder):
    """Wraps the model's attention function once; memory_holder['memory'] selects the active KVMemory (or None)."""
    name = 'sdpa'
    original = ALL_ATTENTION_FUNCTIONS[name]

    def wrapped(module, query, key, value, attention_mask, scaling=None, **kwargs):
        memory = memory_holder.get('memory')
        if memory is None:
            return original(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
        return memory.attention(module, query, key, value, attention_mask, scaling, original, **kwargs)
    ALL_ATTENTION_FUNCTIONS[name] = wrapped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='../dataset_files/knowledge/fictional_people_capacity.json')
    parser.add_argument('--save_path', type=str, default='../results/knowledge_capacity_kv_memory')
    parser.add_argument('--heads_from', type=str,
                        default='../findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs/summary.json')
    parser.add_argument('--n_heads', type=int, default=10, help="Number of top heads from --heads_from")
    parser.add_argument('--layers_all_heads', type=int, nargs='+', default=None,
                        help='Instead of top heads: give memory to every head of these layers')
    parser.add_argument('--n_slots', type=int, default=4)
    parser.add_argument('--n_facts', type=int, default=None)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.05)
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
    assert model.config._attn_implementation == 'sdpa', model.config._attn_implementation
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_seed(args.seed)
    check_single_token_answers(data['people'], tokenizer)
    os.makedirs(args.save_path, exist_ok=True)
    holder = {}
    install(holder)

    if args.layers_all_heads:
        heads = [(l, h) for l in args.layers_all_heads for h in range(model_config['n_heads'])]
    else:
        heads = [(int(l), int(h)) for l, h, _ in json.load(open(args.heads_from))['top_heads'][:args.n_heads]]

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

    def run(input_ids, attention_mask, last):
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        return model(input_ids=input_ids, attention_mask=attention_mask).logits[rows, last].float()

    known_batch = encode([bare_prompt(q, a, model_config) for q, a in KNOWN_FACTS], [None] * len(KNOWN_FACTS), tokenizer, model.device)

    rows, curves, learned = [], {}, {}
    for person in people:
        name = person['name']
        train_items, test_items = questions_for(person, 'train'), questions_for(person, 'test')
        train_batch, test_batch = batch_for(train_items, name), batch_for(test_items, name)
        train_targets = torch.tensor([answer_id(a) for _, _, a in train_items], device=model.device)

        memory = KVMemory(heads, args.n_slots, model, model_config)
        holder['memory'] = memory
        optimizer = torch.optim.Adam(list(memory.params.values()), lr=args.lr)
        input_ids, attention_mask, last, _ = train_batch
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = torch.nn.functional.cross_entropy(run(input_ids, attention_mask, last), train_targets)
                (loss * args.loss_scale).backward()
            for p in memory.params.values():
                p.grad /= args.loss_scale
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
            optimizer.step()
            losses.append(loss.item())
        curves[name] = losses
        learned[name] = {str(k): p.detach().cpu() for k, p in memory.params.items()}

        for split, items, batch in [('train', train_items, train_batch), ('test', test_items, test_batch),
                                    ('known_facts', [(q, None, a) for q, a in KNOWN_FACTS], known_batch)]:
            input_ids, attention_mask, last, _ = batch
            with torch.no_grad():
                logits = run(input_ids, attention_mask, last)
            pred_city = [cities[i] for i in logits[:, city_ids].argmax(1).tolist()]
            pred_token = [tokenizer.decode(i).strip() for i in logits.argmax(1).tolist()]
            for (q, r, a), pc, pt in zip(items, pred_city, pred_token):
                rows.append({'person': name, 'split': split, 'relation': r, 'question': q, 'answer': a,
                             'pred_city': pc if split != 'known_facts' else None, 'pred_token': pt})
        holder['memory'] = None
        test_rows = [r for r in rows if r['person'] == name and r['split'] == 'test']
        print(f"{name}: final train loss {losses[-1]:.4f}, held-out accuracy "
              f"{sum(r['pred_city'] == r['answer'] for r in test_rows) / len(test_rows):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df['correct_city'] = df['pred_city'] == df['answer']
    df['correct_token'] = df['pred_token'] == df['answer']
    own = {p['name']: set(p['facts'].values()) for p in people}
    df['own_city_error'] = [(not c) and pc in own[p] for c, pc, p in zip(df['correct_city'], df['pred_city'], df['person'])]
    df.to_csv(f'{args.save_path}/results.csv', index=False)
    torch.save({'memories': learned, 'loss_curves': curves, 'heads': heads}, f'{args.save_path}/memories.pt')

    facts = df[df.split.isin(['train', 'test'])]
    per_person = df[df.split == 'test'].groupby('person').correct_city.mean()
    summary = {'args': vars(args), 'heads': heads, 'numbers_per_person': KVMemory(heads, args.n_slots, model, model_config).n_numbers(),
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
