# Learned vectors per head and per layer on Qwen3-4B (2026-11-12)

The [capacity experiment](../2026-11-12_qwen3-4b_learned_vector_capacity/README.md) showed that a single learned vector holding 8 facts generalises to only 58–77% of held-out phrasings. This tests two ways to spread the learned representation over more of the model:

- **Per head (option 1):** one vector for each of the top-10 heads from the [stable extraction run](../2026-11-12_qwen3-4b_extraction_logprob_otherperson/README.md), added to that head's output. This combines FV head selection with learning.
- **Per layer (option 2):** one full vector for each of several layers, which separates the effect of using several layers from the effect of using selected heads.

## Reproduce

- **Code:** commit `e69cbd769b7c07b6d96afe8e16dc02ed215a7540` on branch `qwen-fv-extraction`
- **Data:** `dataset_files/knowledge/fictional_people_capacity.json` (8 people × 8 city facts, setup-only), all 8 facts per person
- **Heads:** the top 10 heads of the stable extraction run, selected on the other, 6-person dataset: (31, 3), (32, 8), (24, 26), (34, 1), (33, 29), (31, 2), (35, 25), (29, 11), (26, 20), (29, 24) as (layer, head).
- **Commands:** `outputs/run_interventions.sh`. Each run is `learn_knowledge_intervention.py --n_facts 8` with the options in the table below. The default is 150 Adam steps with learning rate 0.1.
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** about 21 minutes for all 7 runs.

| Run | Options |
|---|---|
| `heads_last_lr0.1` | `--mode heads --position last` |
| `heads_last_lr0.5_300steps` | `--mode heads --position last --lr 0.5 --steps 300` |
| `heads_name_lr0.5_300steps` | `--mode heads --position name --lr 0.5 --steps 300` |
| `layers_head_layers_last` | `--mode layers --layers 24 26 29 31 32 33 34 35 --position last` |
| `layers_6_12_18_24_last` | `--mode layers --layers 6 12 18 24 --position last` |
| `layers_6_12_18_24_name` | `--mode layers --layers 6 12 18 24 --position name` |
| `layers_all_last` | `--mode layers --layers 0 … 35 --position last` |

Head vectors learned slowly in a 50-step test (gradients about 10 times smaller), so they also got a tuned run with learning rate 0.5 and 300 steps.

## Setup

- **Head vectors** (128 numbers each) are added to a head's output, before the layer's output projection.
- **Layer vectors** (2,560 numbers each) are added to a layer's output.
- In both cases they are added at one position (the last token or the person's name) and trained like the single learned vector.
- **Tests:** held-out paraphrases (1 per fact, 64 in total), and 4 capital questions for side effects. Capital questions have no name, so `name` interventions are added at their last token there.
- **Comparison:** the single learned vector with 8 facts from the capacity experiment.

## Results

| Intervention | Numbers per person | Held-out right | People with all 8 held-out right | Capitals still right |
|---|---|---|---|---|
| Single vector, last token, layer 6 | 2,560 | 0.69 | 0/8 | 0.81 |
| Single vector, last token, layer 12 | 2,560 | 0.77 | 0/8 | 0.66 |
| Single vector, name, layer 6 | 2,560 | 0.58 | 0/8 | 0.69 |
| Single vector, name, layer 12 | 2,560 | 0.66 | 0/8 | 0.72 |
| 10 heads, last token, learning rate 0.1 | 1,280 | 0.83 | 0/8 | 0.34 |
| 10 heads, last token, tuned | 1,280 | **0.92** | 3/8 | **0.00** |
| 10 heads, name, tuned | 1,280 | 0.56 | 0/8 | 0.00 |
| Layers 6, 12, 18, 24, last token | 10,240 | 0.88 | 1/8 | 0.28 |
| Layers 6, 12, 18, 24, name | 10,240 | 0.72 | 1/8 | 0.09 |
| The heads' 8 layers, last token | 20,480 | **1.00** | 8/8 | **0.00** |
| All 36 layers, last token | 92,160 | 1.00 | 8/8 | 0.00 |

- Training questions are 100% right for every intervention.
- At the last token, every multi-head or multi-layer intervention beats the single vector on held-out questions. With only 1,280 numbers, the tuned head vectors reach 92%.
- At the name, neither heads nor layers beat the single vector by much.

**The most accurate interventions answer every question with the person's cities.** What they answer to the capital questions:

| Intervention | One of the person's cities | Right capital | Something else |
|---|---|---|---|
| 10 heads, last token, learning rate 0.1 | 0.56 | 0.34 | 0.09 |
| 10 heads, last token, tuned | 1.00 | 0.00 | 0.00 |
| 10 heads, name, tuned | 0.12 | 0.00 | 0.88 |
| Layers 6, 12, 18, 24, last token | 0.72 | 0.28 | 0.00 |
| Layers 6, 12, 18, 24, name | 0.28 | 0.09 | 0.62 |
| The heads' 8 layers, last token | 1.00 | 0.00 | 0.00 |
| All 36 layers, last token | 1.00 | 0.00 | 0.00 |

For example, with the heads' 8 layers learned for Arlen Tovik, "What is the capital of France?" is answered "Lisbon", Arlen's home city.

Held-out accuracy by fact type is in `analysis/analyse_heads_and_layers.py`'s output. The "birthplace" paraphrase remains the hardest: 12–50% for the last-token head and early-layer interventions, and 100% only with vectors in the late layers.

## Interpretation

- **My prediction was wrong.** I expected late-layer head vectors to do worse than a single early vector, because a fixed vector at a late layer has little computation after it to act on the question. Instead, 10 late heads reach 92% held-out with half the numbers of a single vector, and vectors in the heads' 8 layers reach 100%.
- **But this accuracy comes from overriding the question, not from knowledge alongside the model's own.** The interventions that do best on held-out questions answer unrelated questions with the person's cities. In late layers, they act as a "question type → city" lookup for this person that fires on every prompt. The held-out test can't tell that apart from knowledge, because every held-out question is about the same person.
- **Head selection works as an efficient place to intervene:** the FV heads are where answers are written, so small vectors there have a large effect. That doesn't make them a good place to store knowledge that should only be used when relevant.
- **For RAG:** a document vector is only applied after retrieval, so overriding unrelated questions matters less than in general use. It still matters when retrieval is wrong or the query mixes the document with general knowledge. Accuracy on held-out questions and preservation of other knowledge need to be measured together.
- **Limits:**
  - Only 4 unrelated questions, all capitals.
  - One held-out phrasing per fact.
  - No training signal for leaving other questions alone.

## What we investigate next

1. **Train with unrelated questions too:** add a loss that keeps the model's answers to other questions unchanged (for example, matching the unmodified model's predictions on general questions). Then compare interventions at equal side effects.
2. **Learned key–value memory for the heads** (option 3): the heads' own queries retrieve from learned entries, so the effect should depend on the question. Running on branch `kv-memory`.
3. **A larger and more varied test of side effects,** including questions about other people.

## Files

- `outputs/<run>/`: `results.csv` (one row per question), `summary.json` and `run.log` for each run, plus `outputs/runner.log` and `outputs/run_interventions.sh`.
- `outputs/<run>/interventions.pt` (git-ignored). SHA-256:
  - `heads_last_lr0.1`: `790d804ae6fe57ef830327cf5540847a4236985ecfbf95632ef67b1b84962a65`
  - `heads_last_lr0.5_300steps`: `b1594aafaf6e5a8ac172e27b15c458c57f2d306c1e368f44235387d9e161df63`
  - `heads_name_lr0.5_300steps`: `afb30b30c1e3c38a51a8fb0720e3f33e778117119cfb5889299d84fa44dd3d60`
  - `layers_6_12_18_24_last`: `3b4b527feb1c27ff9faca52d756421ffc5646348c51f24b18179dd82ade1f11b`
  - `layers_6_12_18_24_name`: `a80433142f989961b3e60cc761529c7aec97bf9903d08a2aeca62340bc60526a`
  - `layers_all_last`: `bbabb7611ba5edabd7a9d82ee1f721ec3d4f38a8c81ba8adfa1e33a63b21e5b1`
  - `layers_head_layers_last`: `9e52219692f9f9083b252611886e654e4dc2e8fdb8c7ffdc57e100e7040cc294`
- `analysis/analyse_heads_and_layers.py`: all tables above. It needs the capacity experiment's outputs for the single-vector rows.
