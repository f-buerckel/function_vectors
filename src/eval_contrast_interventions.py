"""
Harder tests for interventions learned by learn_knowledge_contrast.py, with full greedy answers instead of the first token:

- own held-out paraphrases (does the full answer name the right city?)
- two-hop questions: "In which country was {name} born?" Answering needs the stored city and the model's own knowledge
  of its country, so it tests whether the stored fact is usable in reasoning or only a question -> city mapping.
- held-out paraphrases and two-hop questions about other people who were not used as contrast
- the 4 capital questions and general questions unlike the contrast set (some with city answers, some without)

Every set is also answered by the unmodified model, and the person's own questions with the facts in context (the
person's 8 demonstration questions and answers, as in extraction), as an upper reference.

For key-value memory runs, also records the selected heads' attention share on the memory at the prompt's last token.

Usage (from src/): python eval_contrast_interventions.py <run_dir> [<run_dir> ...] [--device cuda] [--max_people N]
Writes <run_dir>/generation_eval.csv and <run_dir>/generation_eval_summary.json.
"""
import argparse, ast, json

import pandas as pd
import torch

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from inject_knowledge_fv import bare_prompt
from learn_knowledge_vector import KNOWN_FACTS, encode
from learn_knowledge_intervention import Intervention
from learn_kv_memory import install
from learn_knowledge_contrast import KVIntervention

N_NEW_TOKENS = 16

COUNTRY = {'Athens': ['Greece'], 'Boston': ['United States', 'USA', 'U.S.'], 'Cairo': ['Egypt'],
           'Chicago': ['United States', 'USA', 'U.S.'], 'Dublin': ['Ireland'], 'Lima': ['Peru'],
           'Lisbon': ['Portugal'], 'Madrid': ['Spain'], 'Mumbai': ['India'], 'Nairobi': ['Kenya'], 'Oslo': ['Norway'],
           'Prague': ['Czech'], 'Seoul': ['Korea'], 'Sydney': ['Australia'], 'Toronto': ['Canada'], 'Vienna': ['Austria']}

TWO_HOP = {'born_city': "In which country was {name} born?", 'work_city': "In which country does {name} work?",
           'study_city': "In which country did {name} study?", 'home_city': "In which country does {name} live?",
           'married_city': "In which country did {name} get married?",
           'childhood_city': "In which country did {name} grow up?",
           'vacation_city': "In which country does {name} go on vacation?",
           'retire_city': "In which country does {name} plan to retire?"}

# General questions unlike the contrast set; the first group has city answers
GENERAL_CITY = [("In which city is the Golden Gate Bridge?", ["San Francisco"]), ("In which city is the Taj Mahal?", ["Agra"]),
                ("Which city is famous for its canals and gondolas?", ["Venice"]), ("In which city is Big Ben?", ["London"]),
                ("In which city is the Kremlin?", ["Moscow"]), ("Which city is home to Hollywood?", ["Los Angeles"])]
GENERAL_OTHER = [("Who painted the Mona Lisa?", ["Leonardo", "Vinci"]), ("Which planet is known as the Red Planet?", ["Mars"]),
                 ("What is the largest mammal?", ["whale"]), ("Who developed the theory of relativity?", ["Einstein"]),
                 ("What currency is used in Japan?", ["yen"]), ("How many continents are there?", ["seven", "7"]),
                 ("What is the chemical formula of water?", ["H2O", "H₂O"]),
                 ("Who was the first person to walk on the Moon?", ["Armstrong"]),
                 ("What is the tallest mountain in the world?", ["Everest"]),
                 ("Which gas do plants absorb from the air?", ["carbon dioxide", "CO2", "CO₂"]),
                 ("Which instrument has 88 keys?", ["piano"]), ("Who wrote the Odyssey?", ["Homer"]),
                 ("What is the longest river in Africa?", ["Nile"]), ("In which country is Mount Fuji?", ["Japan"])]


def generate(forward, prompts, names, positions_kind, tokenizer, device):
    """Greedy answers. The intervention stays at the prompt's position (its last token or the name)."""
    ids, mask, last, name_pos = encode(prompts, names, tokenizer, device)
    positions = last.clone() if positions_kind == 'last' else name_pos
    rows = torch.arange(len(prompts), device=device)
    current, new = last.clone(), []
    for _ in range(N_NEW_TOKENS):
        token = forward(ids, mask, positions, current).argmax(-1)
        new.append(token)
        ids = torch.cat([ids, torch.full_like(ids[:, :1], tokenizer.pad_token_id)], 1)
        mask = torch.cat([mask, torch.zeros_like(mask[:, :1])], 1)
        current = current + 1
        ids[rows, current], mask[rows, current] = token, 1
    texts = tokenizer.batch_decode(torch.stack(new, 1).tolist())
    return [t.strip().split('\n')[0].strip() for t in texts]


def contains(text, options):
    return any(o.lower() in text.lower() for o in options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs='+')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_people', type=int, default=None, help='for quick tests')
    cli = parser.parse_args()
    run_dirs = cli.run_dirs
    torch.set_grad_enabled(False)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer('Qwen/Qwen3-4B', device=cli.device)
    model.eval()
    set_seed(42)
    holder = {}
    install(holder)  # no effect while holder['memory'] is None
    device = model.device

    def plain(ids, mask, positions, last):
        return model(input_ids=ids, attention_mask=mask).logits[torch.arange(ids.shape[0], device=device), last].float()

    cache = {}  # unmodified and in-context answers don't depend on the run

    for run_dir in run_dirs:
        summary = json.load(open(f"{run_dir}/summary.json"))
        args = summary['args']
        data = json.load(open(args['data_path']))
        templates = data['templates']
        people = [{'name': p['name'], 'facts': dict(list(p['facts'].items())[:args['n_facts']])} for p in data['people']]
        saved = torch.load(f"{run_dir}/interventions.pt")['interventions']
        heads = [(int(l), int(h)) for l, h, _ in json.load(open(args['heads_from']))['top_heads'][:args['n_heads']]]
        position = 'last' if args['mode'] == 'kv' else args['position']

        def held_out(person):
            return [(templates[r]['test'].format(name=person['name']), [a], r) for r, a in person['facts'].items()]

        def two_hop(person):
            return [(TWO_HOP[r].format(name=person['name']), COUNTRY[a], r) for r, a in person['facts'].items()]

        def in_context(person, question):
            demos = {'input': [templates[r]['demo'].format(name=person['name']) for r in person['facts']],
                     'output': list(person['facts'].values())}
            prompt_data = word_pairs_to_prompt_data(demos, query_target_pair={'input': question, 'output': 'x'},
                                                    prepend_bos_token=not model_config['prepend_bos'])
            return create_prompt(prompt_data)

        rows = []
        for i, person in enumerate(people[:cli.max_people]):
            name = person['name']
            others = people[i + 1:] + people[:i]
            held_out_people = others[args['n_contrast_people']:]
            if args['mode'] == 'kv':
                intervention = KVIntervention(heads, args['n_slots'], model, model_config, holder)
            else:
                intervention = Intervention(args['mode'], heads if args['mode'] == 'heads' else args['layers'], model, model_config)
            for key, value in saved[name].items():
                k = ast.literal_eval(key)
                intervention.params[k].data.copy_(value.to(device))

            sets = [('own_held_out', held_out(person), name), ('own_two_hop', two_hop(person), name)]
            sets += [('other_held_out', [it for p in held_out_people for it in held_out(p)], None),
                     ('other_two_hop', [it for p in held_out_people for it in two_hop(p)], None),
                     ('capitals', [(q, [a], None) for q, a in KNOWN_FACTS], None),
                     ('general_city', [(q, a, None) for q, a in GENERAL_CITY], None),
                     ('general_other', [(q, a, None) for q, a in GENERAL_OTHER], None)]
            owner_cities = list(person['facts'].values())
            owner_countries = [c for city in owner_cities for c in COUNTRY[city]]
            for kind, items, asked in sets:
                if kind.startswith('other'):  # name of the person asked about, for name-position interventions
                    names = [next(p['name'] for p in held_out_people if p['name'] in q) for q, _, _ in items]
                else:
                    names = [asked] * len(items)
                prompts = [bare_prompt(q, 'x', model_config) for q, _, _ in items]
                shares = [float('nan')] * len(items)
                if args['mode'] == 'kv':  # attention share on the memory at the prompt's last token, first step
                    ids, mask, last, _ = encode(prompts, names, tokenizer, device)
                    intervention.memory.record = {}
                    intervention.run(ids, mask, last, last)
                    r = torch.arange(len(items), device=device)
                    rec = intervention.memory.record
                    shares = torch.stack([rec[l][r, :, last][:, [h for (ll, h) in heads if ll == l]].mean(1)
                                          for l in sorted({l for l, _ in heads})], 1).mean(1).tolist()
                    intervention.memory.record = None
                modified = generate(intervention.run, prompts, names, position, tokenizer, device)
                key = (kind, tuple(prompts))
                if key not in cache:
                    cache[key] = generate(plain, prompts, names, 'last', tokenizer, device)
                unmodified = cache[key]
                context = [None] * len(items)
                if kind.startswith('own'):
                    cprompts = [in_context(person, q) for q, _, _ in items]
                    ckey = (kind, tuple(cprompts))
                    if ckey not in cache:
                        cache[ckey] = generate(plain, cprompts, [None] * len(items), 'last', tokenizer, device)
                    context = cache[ckey]
                for (q, ref, rel), m, u, c, s in zip(items, modified, unmodified, context, shares):
                    rows.append({'person': name, 'set': kind, 'relation': rel, 'question': q, 'reference': '|'.join(ref),
                                 'modified': m, 'unmodified': u, 'in_context': c, 'memory_attention': s,
                                 'modified_right': contains(m, ref), 'unmodified_right': contains(u, ref),
                                 'in_context_right': contains(c, ref) if c is not None else None,
                                 'unchanged': m == u,
                                 'modified_has_owner_city': contains(m, owner_cities),
                                 'modified_has_owner_country': contains(m, owner_countries)})
            print(f"{run_dir}: {name} done", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(f"{run_dir}/generation_eval.csv", index=False)
        cols = ['modified_right', 'unmodified_right', 'in_context_right', 'unchanged', 'modified_has_owner_city',
                'modified_has_owner_country', 'memory_attention']
        table = df.groupby('set')[cols].agg(lambda x: pd.to_numeric(x, errors='coerce').mean())
        with open(f"{run_dir}/generation_eval_summary.json", 'w') as f:
            json.dump(table.to_dict(orient='index'), f, indent=2, default=float)
        pd.set_option('display.width', 220)
        print(run_dir)
        print(table.round(3).to_string())
