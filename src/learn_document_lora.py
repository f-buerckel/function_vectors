"""
PRAG baseline: one LoRA per wiki2026 document, evaluated exactly like the learned interventions
(learn_document_intervention.py): test phrasings of trained facts, held-out facts, two-step questions, other documents'
questions and general questions, all without the document in context.

Reimplementation of PRAG's parametric knowledge injection (Su et al., 2025; settings taken from the official code,
github.com/oneal2000/PRAG, src/encode.py): LoRA on the feed-forward projections (gate_proj, up_proj, down_proj) of every
layer, rank 2, alpha 32, dropout 0, AdamW with learning rate 3e-4, batch size 1, no scheduler.

Training data (--data):
- 'prag': PRAG's examples from augment_documents_prag.py. For each passage with 3 generated question-answer pairs:
  pairs 1 and 2 each with the passage and with its rewrite in context, pair 3 without any passage (5 examples per
  passage). Chat template; the assistant's reply is "The answer is <answer>."; as in PRAG, the loss covers the whole
  sequence (instruction, passages, question and reply), not only the answer.
- 'questions': the same data as the learned interventions: the subagent-written training questions in the Q:/A:
  format (loss on the answer tokens) and, with --doc_lm_weight > 0, the document text in chunks of --chunk_tokens
  (next-token loss on every token).

Differences from PRAG, on purpose: one LoRA per document (trained on all its passages) instead of one per 100-word
passage; float16 base weights with float32 LoRA parameters and loss scaling instead of float32 (16 GB GPU); optional
contrast training (KL to the unmodified model on general and other documents' questions, as for the interventions).

Evaluation formats (--eval_formats): 'qa' is the interventions' Q:/A: prompt; 'chat' is PRAG's chat prompt without
passages, where the model replies "The answer is ...". Both are scored the same way (first line contains the answer).

Usage (from src/): python learn_document_lora.py --data prag --save_path ../results/<run>
"""
import argparse, json, os, random

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from inject_knowledge_fv import bare_prompt
from learn_knowledge_vector import encode
from learn_knowledge_contrast import GENERAL_TRAIN_QUESTIONS
from learn_document_intervention import generate
from document_data import load_documents, make_train_items, evaluate_document, summarise

INSTRUCTION = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge."


def prag_user_message(question, passages=()):
    """PRAG-style user message: instruction, numbered passages (if any), question."""
    parts = [INSTRUCTION] + [f"Passage {k + 1}: {p}" for k, p in enumerate(passages)] + [f"Question: {question}"]
    return '\n\n'.join(parts)


def prag_reply(answer):
    answer = answer.strip()
    return f"The answer is {answer}" + ('' if answer.endswith('.') else '.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B')
    parser.add_argument('--data_dir', type=str, default='../dataset_files/knowledge/wiki2026')
    parser.add_argument('--questions', type=str, default='filtered_v2.json')
    parser.add_argument('--augmented', type=str, default='prag_augmented.json')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--data', type=str, required=True, choices=['prag', 'questions'])
    parser.add_argument('--doc_lm_weight', type=float, default=0.0, help="--data questions: weight of the document-text examples")
    parser.add_argument('--chunk_tokens', type=int, default=256)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--contrast_weight', type=float, default=0.0)
    parser.add_argument('--contrast_per_step', type=int, default=16, help='half general, half other documents')
    parser.add_argument('--n_contrast_docs', type=int, default=6)
    parser.add_argument('--eval_formats', type=str, nargs='+', default=['qa', 'chat'], choices=['qa', 'chat'])
    parser.add_argument('--max_new_tokens', type=int, default=48)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--max_docs', type=int, default=None, help='for quick tests')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    base, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    for p in base.parameters():
        p.requires_grad_(False)
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['down_proj', 'gate_proj', 'up_proj'],
                                            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0))
    model.eval()  # no dropout anywhere; LoRA parameters still get gradients
    lora_params = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in lora_params)
    n_numbers = sum(p.numel() for p in lora_params)
    init_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}  # same start for every document
    inner = model.get_base_model()
    device = base.device

    docs, data = load_documents(args.data_dir, args.questions)
    augmented = {a['doc_id']: a for a in json.load(open(f'{args.data_dir}/{args.augmented}'))} if args.data == 'prag' else None
    train_items = make_train_items('prag' if args.data == 'prag' else 'questions', augmented)
    qa_prompt = lambda q: bare_prompt(q, 'x', model_config)
    chat_prompt = lambda q: tokenizer.apply_chat_template([{'role': 'user', 'content': prag_user_message(q)}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)

    def chat_ids(user, reply):
        prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        return tokenizer(prefix + reply, add_special_tokens=False).input_ids

    def training_examples(q):
        """[(input_ids, labels, weight)], labels -100 where there is no loss."""
        examples = []
        if args.data == 'prag':
            for p in augmented[q['doc_id']]['passages']:
                if len(p['qa']) < 3:
                    continue
                for k, a in enumerate(p['qa'][:3]):
                    contexts = [[p['text']], [p['rewrite']]] if k < 2 else [[]]
                    for ctx in contexts:
                        ids = chat_ids(prag_user_message(a['question'], ctx), prag_reply(a['answer']))
                        examples.append((ids, list(ids), 1.0))
            return examples
        for t, a, _, _ in train_items(q):
            p_ids = tokenizer(qa_prompt(t)).input_ids
            a_ids = tokenizer(' ' + a, add_special_tokens=False).input_ids
            examples.append((p_ids + a_ids, [-100] * len(p_ids) + a_ids, 1.0))
        if args.doc_lm_weight > 0:
            doc = docs[q['doc_id']]
            token_ids = tokenizer(f"{doc['title']}\n\n{doc['text']}").input_ids
            for k in range(0, len(token_ids), args.chunk_tokens):
                c = token_ids[k:k + args.chunk_tokens]
                if len(c) > 1:
                    examples.append((c, list(c), args.doc_lm_weight))
        return examples

    def last_logits(ids, mask, positions, last):
        """float32 logits at each prompt's last token, with whatever adapter state is active."""
        hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return inner.lm_head(hidden[torch.arange(len(last), device=device), last]).float()

    def plain(ids, mask, positions, last):
        with torch.no_grad(), model.disable_adapter():
            return last_logits(ids, mask, positions, last)

    general_batch = encode([qa_prompt(t) for t in GENERAL_TRAIN_QUESTIONS], [None] * len(GENERAL_TRAIN_QUESTIONS), tokenizer, device)
    general_ref = torch.log_softmax(plain(*general_batch[:3], general_batch[2]), -1) if args.contrast_weight > 0 else None
    kl = lambda new, ref: torch.nn.functional.kl_div(new, ref, log_target=True, reduction='batchmean')
    unmodified_cache, rows, curves, learned = {}, [], {}, {}
    for i, q in enumerate(data[:args.max_docs]):
        doc_id = q['doc_id']
        others = data[i + 1:] + data[:i]
        contrast_docs, held_out_docs = others[:args.n_contrast_docs], others[args.n_contrast_docs:]
        set_peft_model_state_dict(model, init_state)
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
        examples = training_examples(q)
        rng = random.Random(args.seed + i)
        if args.contrast_weight > 0:
            other_qs = [t for c in contrast_docs for t, _, _, _ in train_items(c)]
            other_batch = encode([qa_prompt(t) for t in other_qs], [None] * len(other_qs), tokenizer, device)
            other_ref = torch.log_softmax(plain(*other_batch[:3], other_batch[2]), -1)
        losses = []
        for epoch in range(args.epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for k in order:
                ids, labels, weight = examples[k]
                optimizer.zero_grad()
                with torch.enable_grad():
                    x = torch.tensor([ids], device=device)
                    logits = inner(input_ids=x).logits[0, :-1].float()
                    own = weight * torch.nn.functional.cross_entropy(logits, torch.tensor(labels[1:], device=device),
                                                                     ignore_index=-100)
                    scaler.scale(own).backward()
                    del logits
                    contrast = 0.0
                    if args.contrast_weight > 0:
                        n = args.contrast_per_step // 2
                        gi = torch.tensor(rng.sample(range(len(GENERAL_TRAIN_QUESTIONS)), n), device=device)
                        oi = torch.tensor(rng.sample(range(len(other_qs)), min(n, len(other_qs))), device=device)
                        for batch, ref, idx in [(general_batch, general_ref, gi), (other_batch, other_ref, oi)]:
                            new = torch.log_softmax(last_logits(batch[0][idx], batch[1][idx], None, batch[2][idx]), -1)
                            part = kl(new, ref[idx])
                            scaler.scale(args.contrast_weight * part).backward()
                            contrast += part.item()
                scaler.step(optimizer)
                scaler.update()
                losses.append((own.item(), contrast))
        curves[doc_id] = losses
        learned[doc_id] = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
        tail = losses[-len(examples):]
        print(f"{doc_id}: {len(examples)} examples x {args.epochs} epochs, mean own loss of the last epoch "
              f"{sum(o for o, _ in tail) / len(tail):.4f}, contrast {sum(c for _, c in tail) / len(tail):.4f}", flush=True)
        del optimizer, scaler
        torch.cuda.empty_cache()

        with torch.no_grad():
            answer = lambda forward: lambda prompts: generate(forward, prompts, tokenizer, device, args.max_new_tokens,
                                                              args.eval_batch_size)
            for fmt in args.eval_formats:
                rows += evaluate_document(q, held_out_docs, train_items, qa_prompt if fmt == 'qa' else chat_prompt,
                                          answer(last_logits), answer(plain), unmodified_cache, extra={'format': fmt})

    df = pd.DataFrame(rows)
    df.to_csv(f'{args.save_path}/results.csv', index=False)
    torch.save({'loras': learned, 'loss_curves': curves}, f'{args.save_path}/loras.pt')
    summary = {'args': vars(args), 'numbers_per_document': n_numbers, 'n_docs': len(curves),
               'n_examples_per_document': {d: len(c) // args.epochs for d, c in curves.items()},
               'formats': {fmt: summarise(df[df.format == fmt]) for fmt in args.eval_formats}}
    with open(f'{args.save_path}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'args'}, indent=2, default=float))
