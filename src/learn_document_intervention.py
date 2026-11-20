"""
Learns one intervention per document (wiki2026 dataset) so the frozen model answers questions about it without the
document in context. Like learn_knowledge_contrast.py, but for real documents:

- multi-token answers: the training input is the question followed by the answer, and the loss is the cross-entropy of
  every answer token (teacher forcing); the intervention sits at the question's last token ("A:"), or in the heads'
  key-value memory, which is visible from every position;
- contrast: other documents' training questions and general questions, with a KL loss to the unmodified model at the
  answer position (first answer token only);
- evaluation by greedy generation: an answer counts as right if it contains the reference answer or an alias.

Splits per document i (documents ordered by id): the next --n_contrast_docs documents are contrast documents, the
remaining ones are held out and only used to test side effects.

Usage (from src/): python learn_document_intervention.py --mode heads --lr 0.5 --save_path ../results/<run> [--fast]
"""
import argparse, json, os

import pandas as pd
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from inject_knowledge_fv import bare_prompt
from learn_knowledge_vector import KNOWN_FACTS, encode
from learn_knowledge_intervention import Intervention, PrefixCache
from learn_knowledge_contrast import GENERAL_TRAIN_QUESTIONS, KVIntervention, converged
from learn_kv_memory import install
from eval_contrast_interventions import GENERAL_CITY, GENERAL_OTHER
from filter_wiki2026_questions import is_right


def encode_with_answers(prompts, answers, tokenizer, device):
    """
    Right-padded batch of prompt + ' ' + answer. Returns input_ids, attention_mask, the prompt's last position,
    out_positions (B, K): the positions whose logits predict the K answer tokens, targets (B, K) with -100 as padding.
    """
    seqs, prompt_lens, answer_ids = [], [], []
    for p, a in zip(prompts, answers):
        p_ids = tokenizer(p).input_ids
        a_ids = tokenizer(' ' + a, add_special_tokens=False).input_ids
        seqs.append(p_ids + a_ids)
        prompt_lens.append(len(p_ids))
        answer_ids.append(a_ids)
    T, K = max(map(len, seqs)), max(map(len, answer_ids))
    input_ids = torch.full((len(seqs), T), tokenizer.pad_token_id)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    out_positions = torch.zeros((len(seqs), K), dtype=torch.long)
    targets = torch.full((len(seqs), K), -100)
    for i, (s, P, a) in enumerate(zip(seqs, prompt_lens, answer_ids)):
        input_ids[i, :len(s)], mask[i, :len(s)] = torch.tensor(s), 1
        out_positions[i, :len(a)] = torch.arange(P - 1, P - 1 + len(a))
        out_positions[i, len(a):] = P - 1  # padding; ignored through targets
        targets[i, :len(a)] = torch.tensor(a)
    last = torch.tensor([P - 1 for P in prompt_lens])
    return [t.to(device) for t in (input_ids, mask, last, out_positions, targets)]


def generate(forward, prompts, tokenizer, device, n_new, batch_size):
    """Greedy answers with the intervention at each prompt's last token (no cache; first line of the answer)."""
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        ids, mask, last, _ = encode(chunk, [None] * len(chunk), tokenizer, device)
        positions, rows, current, new = last.clone(), torch.arange(len(chunk), device=device), last.clone(), []
        for _ in range(n_new):
            token = forward(ids, mask, positions, current).argmax(-1)
            new.append(token)
            ids = torch.cat([ids, torch.full_like(ids[:, :1], tokenizer.pad_token_id)], 1)
            mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], 1)
            current = current + 1
            ids[rows, current], mask[rows, current] = token, 1
        texts = tokenizer.batch_decode(torch.stack(new, 1).tolist(), skip_special_tokens=True)
        out += [t.strip().split('\n')[0].strip() for t in texts]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_dir', type=str, default='../dataset_files/knowledge/wiki2026')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['heads', 'layers', 'kv'])
    parser.add_argument('--layers', type=int, nargs='+', default=None, help='layers mode')
    parser.add_argument('--heads_from', type=str,
                        default='../findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs/summary.json')
    parser.add_argument('--n_heads', type=int, default=10)
    parser.add_argument('--n_slots', type=int, default=4, help='kv mode')
    parser.add_argument('--contrast_weight', type=float, default=1.0)
    parser.add_argument('--n_contrast_docs', type=int, default=6)
    parser.add_argument('--other_questions_per_step', type=int, default=24)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--loss_scale', type=float, default=1024.0)
    parser.add_argument('--fast', action='store_true', help='PrefixCache for the frozen layers (same results)')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=48)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--max_docs', type=int, default=None, help='for quick tests')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    holder = {}
    if args.mode == 'kv':
        install(holder)
    device = model.device

    docs = {d['doc_id']: d for d in map(json.loads, open(f'{args.data_dir}/documents.jsonl'))}
    data = [q for q in json.load(open(f'{args.data_dir}/filtered.json')) if q['facts']]
    data = sorted(data, key=lambda q: q['doc_id'])
    heads = [(int(l), int(h)) for l, h, _ in json.load(open(args.heads_from))['top_heads'][:args.n_heads]]

    def make_intervention():
        if args.mode == 'kv':
            return KVIntervention(heads, args.n_slots, model, model_config, holder)
        return Intervention(args.mode, heads if args.mode == 'heads' else args.layers, model, model_config)

    prompt = lambda q: bare_prompt(q, 'x', model_config)
    train_items = lambda q: [(t, f['answer']) for f in q['facts'] for t in f['train']]
    def test_items(q):
        items = [(f['test'], [f['answer']] + f.get('aliases', []), 'test', f['fact_id']) for f in q['facts']]
        return items + [(h['question'], [h['answer']] + h.get('aliases', []), 'two_hop', h['q_id']) for h in q['two_hop']]
    general_pairs = [(q, [a]) for q, a in KNOWN_FACTS] + [(q, list(a)) for q, a in GENERAL_CITY + GENERAL_OTHER]
    general_test = [(q, answers, 'general', f'g{i}') for i, (q, answers) in enumerate(general_pairs)]

    def plain(ids, mask, positions, last):
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask).logits[torch.arange(ids.shape[0], device=device), last].float()

    general_batch = encode([prompt(q) for q in GENERAL_TRAIN_QUESTIONS], [None] * len(GENERAL_TRAIN_QUESTIONS), tokenizer, device)
    unmodified_cache = {}
    rows, curves, learned = [], {}, {}
    for i, q in enumerate(data[:args.max_docs]):
        doc_id = q['doc_id']
        others = data[i + 1:] + data[:i]
        contrast_docs, held_out_docs = others[:args.n_contrast_docs], others[args.n_contrast_docs:]

        items = train_items(q)
        ids, mask, last, out_pos, targets = encode_with_answers([prompt(t) for t, _ in items], [a for _, a in items],
                                                                tokenizer, device)
        other_qs = [t for c in contrast_docs for t, _ in train_items(c)]
        other_batch = encode([prompt(t) for t in other_qs], [None] * len(other_qs), tokenizer, device)

        intervention = make_intervention()
        caches = {'own': None, 'general': None, 'other': None}
        if args.fast:
            start = intervention.start_layer()
            caches = {'own': PrefixCache(model, ids, mask, start),
                      'general': PrefixCache(model, general_batch[0], general_batch[1], start),
                      'other': PrefixCache(model, other_batch[0], other_batch[1], start)}
        general_ref = torch.log_softmax(plain(*general_batch[:3], general_batch[2]), -1)
        other_ref = torch.log_softmax(plain(*other_batch[:3], other_batch[2]), -1)

        optimizer = torch.optim.Adam(list(intervention.params.values()), lr=args.lr)
        generator = torch.Generator().manual_seed(args.seed + i)
        kl = lambda new, ref: torch.nn.functional.kl_div(new, ref, log_target=True, reduction='batchmean')
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                logits = intervention.run(ids, mask, last, last, cache=caches['own'], out_positions=out_pos)
                own_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                             ignore_index=-100)
                loss, contrast_loss = own_loss, torch.tensor(0.0)
                if args.contrast_weight > 0:
                    idx = torch.randperm(len(other_qs), generator=generator)[:args.other_questions_per_step].to(device)
                    sub = [t[idx] for t in other_batch]
                    new_general = torch.log_softmax(intervention.run(general_batch[0], general_batch[1], general_batch[2],
                                                                     general_batch[2], cache=caches['general']), -1)
                    new_other = torch.log_softmax(intervention.run(sub[0], sub[1], sub[2], sub[2], cache=caches['other'],
                                                                   cache_rows=idx), -1)
                    contrast_loss = kl(new_general, general_ref) + kl(new_other, other_ref[idx])
                    loss = own_loss + args.contrast_weight * contrast_loss
                (loss * args.loss_scale).backward()
            for p in intervention.params.values():
                p.grad /= args.loss_scale
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {doc_id}"
            optimizer.step()
            losses.append((own_loss.item(), float(contrast_loss)))
            if args.early_stop and converged([o + args.contrast_weight * c for o, c in losses]):
                break
        curves[doc_id] = losses
        learned[doc_id] = {str(k): p.detach().cpu() for k, p in intervention.params.items()}
        print(f"{doc_id}: {len(losses)} steps, final own loss {losses[-1][0]:.4f}, contrast loss {losses[-1][1]:.4f}", flush=True)

        # Evaluation: own training and test questions, held-out documents' test questions, general questions
        eval_sets = [('train', [(t, [f['answer']] + f.get('aliases', []), 'train', f"{f['fact_id']}_train{j}")
                                for f in q['facts'] for j, t in enumerate(f['train'])], q)]
        eval_sets += [('own', test_items(q), q)]
        eval_sets += [('held_out_doc', test_items(h), h) for h in held_out_docs]
        eval_sets += [('general', general_test, None)]
        own_answers = [f['answer'] for f in q['facts']]
        with torch.no_grad():
            for split, its, asked_doc in eval_sets:
                prompts = [prompt(t) for t, _, _, _ in its]
                modified = generate(lambda a, b, c, d: intervention.run(a, b, c, d), prompts, tokenizer, device,
                                    args.max_new_tokens, args.eval_batch_size)
                missing = [p for p in prompts if p not in unmodified_cache]
                if missing:
                    for p, text in zip(missing, generate(plain, missing, tokenizer, device, args.max_new_tokens,
                                                         args.eval_batch_size)):
                        unmodified_cache[p] = text
                for (t, answers, kind, item_id), p, m in zip(its, prompts, modified):
                    u = unmodified_cache[p]
                    rows.append({'doc_id': doc_id, 'split': split, 'asked_doc': asked_doc['doc_id'] if asked_doc else None,
                                 'kind': kind, 'item_id': item_id, 'question': t, 'answers': '|'.join(answers),
                                 'modified': m, 'unmodified': u, 'modified_right': is_right(m, answers),
                                 'unmodified_right': is_right(u, answers), 'unchanged': m == u,
                                 'gives_own_answer': is_right(m, own_answers) and split != 'own' and split != 'train'})

    df = pd.DataFrame(rows)
    df.to_csv(f'{args.save_path}/results.csv', index=False)
    torch.save({'interventions': learned, 'loss_curves': curves}, f'{args.save_path}/interventions.pt')
    g = lambda split, kind=None: df[(df.split == split) & ((df.kind == kind) if kind else True)]
    summary = {'args': vars(args), 'numbers_per_document': make_intervention().n_numbers(), 'n_docs': len(curves),
               'train_right': g('train').modified_right.mean(),
               'test_right': g('own', 'test').modified_right.mean(),
               'two_hop_right': g('own', 'two_hop').modified_right.mean(),
               'test_right_unmodified': g('own', 'test').unmodified_right.mean(),
               'two_hop_right_unmodified': g('own', 'two_hop').unmodified_right.mean(),
               'held_out_docs_unchanged': g('held_out_doc').unchanged.mean(),
               'held_out_docs_gives_own_answer': g('held_out_doc').gives_own_answer.mean(),
               'held_out_docs_right': g('held_out_doc').modified_right.mean(),
               'general_right': g('general').modified_right.mean(),
               'general_right_unmodified': g('general').unmodified_right.mean(),
               'general_unchanged': g('general').unchanged.mean(),
               'mean_steps': sum(len(c) for c in curves.values()) / len(curves),
               'final_own_loss_max': max(c[-1][0] for c in curves.values())}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'args'}, indent=2, default=float))
