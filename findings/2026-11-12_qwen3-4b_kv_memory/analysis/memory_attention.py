"""
How much do the selected heads attend to the learned memory, depending on the question? Also: what does a person's
memory make the model answer for other people's questions and for capitals?

Needs the GPU and the model. Usage (from function_vectors/src):
python ../findings/2026-11-12_qwen3-4b_kv_memory/analysis/memory_attention.py ../findings/2026-11-12_qwen3-4b_kv_memory/outputs/top10_heads_4_slots
"""
import ast, json, sys

import pandas as pd
import torch

sys.path.insert(0, '.')
from utils.model_utils import load_gpt_model_and_tokenizer
from learn_kv_memory import KVMemory, install
from learn_knowledge_vector import KNOWN_FACTS, encode
from inject_knowledge_fv import bare_prompt

run_dir = sys.argv[1]
summary = json.load(open(f"{run_dir}/summary.json"))
saved = torch.load(f"{run_dir}/memories.pt")
data = json.load(open('../dataset_files/knowledge/fictional_people_capacity.json'))
people, templates = data['people'], data['templates']
heads = [tuple(h) for h in summary['heads']]

torch.set_grad_enabled(False)
model, tokenizer, cfg = load_gpt_model_and_tokenizer('Qwen/Qwen3-4B', device='cuda')
holder = {}
install(holder)
cities = sorted({c for p in people for c in p['facts'].values()})
city_ids = torch.tensor([tokenizer(' ' + c).input_ids[0] for c in cities], device=model.device)

def held_out(person):
    return [(templates[r]['test'].format(name=person['name']), person['facts'][r]) for r in person['facts']]

def run(items, names, memory):
    ids, mask, last, _ = encode([bare_prompt(q, a, cfg) for q, a in items], names, tokenizer, model.device)
    memory.record = {}
    logits = model(input_ids=ids, attention_mask=mask).logits[torch.arange(len(items)), last].float()
    rows = torch.arange(len(items), device=model.device)
    # Mean over the selected heads of their attention share on the memory at each prompt's last token
    shares = torch.stack([memory.record[l][rows, :, last][:, [h for (ll, h) in heads if ll == l]].mean(1)
                          for l in sorted({l for l, _ in heads})], 1).mean(1)
    pred_city = [cities[i] for i in logits[:, city_ids].argmax(1).tolist()]
    pred_token = [tokenizer.decode(i).strip() for i in logits.argmax(1).tolist()]
    return shares.tolist(), pred_city, pred_token

rows = []
for i, person in enumerate(people):
    memory = KVMemory(heads, summary['args']['n_slots'], model, cfg)
    for key_str, value in saved['memories'][person['name']].items():
        memory.params[ast.literal_eval(key_str)].data.copy_(value.to(model.device))
    holder['memory'] = memory
    other = people[(i + 1) % len(people)]
    sets = [('own held-out questions', held_out(person), person['name']),
            ("another person's held-out questions", held_out(other), other['name']),
            ('capital questions', list(KNOWN_FACTS), None)]
    for kind, items, asked in sets:
        shares, pred_city, pred_token = run(items, [asked] * len(items), memory)
        for (q, a), s, pc, pt in zip(items, shares, pred_city, pred_token):
            rows.append({'memory_of': person['name'], 'questions': kind, 'question': q, 'answer': a, 'memory_attention': s,
                         'pred_city': pc, 'pred_token': pt, 'answer_right': pt == a,
                         'answer_is_memory_owner_city': pt in set(person['facts'].values())})
    holder['memory'] = None

df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/memory_attention.csv", index=False)
pd.set_option('display.width', 200)
print(df.groupby('questions')[['memory_attention', 'answer_right', 'answer_is_memory_owner_city']].mean().round(3))
