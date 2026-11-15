# Contrast training for learned interventions on Qwen3-4B (2026-11-15)

**Branch `contrast-training`.** Every learned intervention so far was trained only on its owner's questions. The most accurate ones ([head and layer vectors](../2026-11-12_qwen3-4b_learned_heads_and_layers/README.md), [key–value memory](../2026-11-12_qwen3-4b_kv_memory/README.md)) answered every question with the owner's cities, so their held-out accuracy partly measured an override. This adds **contrast questions**: questions where the intervention is active but must leave the model's prediction unchanged. Five interventions are trained without (λ = 0) and with (λ = 1) contrast and measured the same way.

**Relation to "Contrastive ICL"** in the RAG best-practices paper (Li et al. 2025, *Enhancing Retrieval-Augmented Generation: A Study of Best Practices*): not the same thing. There, contrastive in-context learning puts demonstrations with correct and incorrect answers into the prompt at inference time. Here, contrast is a training loss on the intervention. It is closer to the locality losses used in knowledge editing, which keep the edited model's predictions on unrelated prompts close to the original model's.

## Reproduce

- **Code:** commit `5344331f06078aa0db2b2c197ffd336465de3148` on branch `contrast-training`, script `src/learn_knowledge_contrast.py`.
- **Data:** `dataset_files/knowledge/fictional_people_capacity.json` (8 people × 8 city facts, setup-only), all 8 facts.
- **Heads:** the top-10 heads of the [stable extraction run](../2026-11-12_qwen3-4b_extraction_logprob_otherperson/README.md), as in experiments 8 and 9.
- **Commands:** `outputs/run_contrast.sh`. Every run uses 300 Adam steps and loss scale 1024; λ is `--contrast_weight 0` or `1`.

| Run | Options | Same intervention as |
|---|---|---|
| `vector_layer12_last_w*` | `--mode layers --layers 12 --position last` (learning rate 0.1) | experiment 7, single vector |
| `vector_layer12_name_w*` | `--mode layers --layers 12 --position name` (learning rate 0.1) | experiment 7, single vector |
| `heads_last_w*` | `--mode heads --position last --lr 0.5` | experiment 8, tuned head vectors |
| `kv_top10_4slots_w*` | `--mode kv --lr 0.05 --n_slots 4` | experiment 9 |
| `layers_head_layers_last_w*` | `--mode layers --layers 24 26 29 31 32 33 34 35 --position last` (learning rate 0.1) | experiment 8, the heads' 8 layers |

- **Analysis**, from the repo root: `python findings/2026-11-15_qwen3-4b_contrast_training/analysis/analyse_contrast.py findings/2026-11-15_qwen3-4b_contrast_training/outputs` (needs pandas only).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16 model (float32 attention in memory layers for `kv`).
- **Runtime:** 86 minutes for all 10 runs (λ = 0 runs 4–5 minutes each, λ = 1 runs 11–14 minutes).

## Setup

- **Loss:** cross-entropy on the owner's 24 training questions (8 facts × 3 phrasings) + λ × contrast loss.
- **Contrast loss:** KL divergence from the unmodified model's next-token distribution to the modified one, summed over two sets per step, with the intervention active:
  - 24 general questions (other countries' capitals, landmarks, trivia; `GENERAL_TRAIN_QUESTIONS`);
  - 24 questions sampled from the training questions of 4 other people (the next 4 in the dataset).
- **Interventions at the name** are added at the last token when a question has no name, as before.
- **Tests,** none of them used in training:
  - the owner's held-out paraphrases (1 per fact, 64 in total; bag-of-answers baseline 12.5%);
  - the 4 capital questions (France, Germany, Italy, Japan; the unmodified model gets all of them right);
  - held-out paraphrases about the **3 other people not used as contrast** (192 questions in total). The unmodified model doesn't know these people.
- **"Unchanged"** compares only the top first token with the unmodified model. KL is over the full next-token distribution.

## Results

| Intervention | Numbers per person | λ | Held-out right | Capitals right | Capitals answered with one of the 16 dataset cities | Other people answered with owner's city | Other people unchanged | KL, other people | KL, capitals |
|---|---|---|---|---|---|---|---|---|---|
| Single vector, layer 12, last token | 2,560 | 0 | 0.72 | 0.66 | 0.13 | 0.89 | 0.01 | 15.5 | 5.06 |
| | | 1 | 0.58 | 0.88 | 0.00 | 0.01 | 0.77 | 0.23 | 0.025 |
| Single vector, layer 12, name | 2,560 | 0 | 0.66 | 0.75 | 0.00 | 0.45 | 0.06 | 7.9 | 0.56 |
| | | 1 | 0.59 | 0.88 | 0.00 | 0.00 | **0.91** | **0.033** | **0.012** |
| 10 head vectors, last token | 1,280 | 0 | 0.92 | 0.00 | 1.00 | 1.00 | 0.00 | 16.6 | 8.82 |
| | | 1 | **0.86** | 0.78 | 0.00 | 0.01 | 0.73 | 0.37 | 0.21 |
| Memory, top-10 heads, 4 entries | 10,250 | 0 | 0.88 | 0.00 | 0.91 | 0.99 | 0.01 | 12.1 | 7.29 |
| | | 1 | 0.75 | 0.88 | 0.00 | 0.01 | 0.86 | 0.15 | **0.012** |
| The heads' 8 layers, last token | 20,480 | 0 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 28.8 | 15.1 |
| | | 1 | 0.84 | 0.69 | 0.00 | 0.00 | 0.73 | 0.33 | 0.11 |

- Training questions are 100% right in every run (largest final loss 0.007).
- **Cross-check:** the λ = 0 single vector at the last token reproduces experiment 7 (held-out 0.72 with 300 steps against 0.77 with 150; capitals 0.66 in both).
- **Every held-out error is another of the owner's cities**, with or without contrast (5–27 errors per run).

**What still changes with contrast.** The "unchanged" rates of 73–91% are not leftover city answers:

- On other people's questions, only 0–1% of answers become one of the owner's cities (at most 2 of 192), and none becomes any other of the 16 dataset cities. The other changes swap one non-answer first token for another, for example "Hal" → "The".
- On capitals, every change is "The" or "A" in place of the city, for example "What is the capital of France?" → "The" for 4 of the 8 people's KV memories. The single first token doesn't show whether the full answer would still be right.

**Held-out accuracy by fact** (from the analysis script): the "born" paraphrase stays the hardest, 12–25% for every last-token intervention with contrast. With contrast, the name vector is the exception (62%), but it gets "home" and "work" only 12% right.

## Interpretation

- **Contrast training removes the override.** With λ = 1, no intervention answers another person's questions with the owner's cities (at most 1%, from 45–100%), none answers a capital with a dataset city (from up to 100%), and the KL to the unmodified model drops by 40–600 times. This generalises to people who were never contrast examples and to held-out phrasings, so the interventions learned to depend on *who* is asked about, not just on the question type.
- **Part of the earlier held-out accuracy was the override.** With contrast, held-out accuracy falls by 6–16 points: head vectors 92 → 86%, the 8 layers 100 → 84%, KV memory 88 → 75%, single vectors 72 → 58% and 66 → 59%. So the λ = 0 numbers of experiments 7–9 overstate how much knowledge the interventions hold.
- **At equal side effects, the 10 head vectors are the most accurate per number:** 86% held-out with 1,280 numbers, against 84% for the 8 layers with 16 times as many. That's 6.9 times the bag-of-answers baseline. FV head selection is a good place to store binding knowledge once the training data asks for binding.
- **The name vector has the smallest side effects** (91% of other people's first tokens unchanged, KL 0.03) but stays at 59%. The KV memory has the smallest effect on capitals (KL 0.012), but its accuracy fell the most. Whether it now reads its memory depending on the question isn't measured yet.
- **For RAG:** a document intervention trained with contrast behaves more like a document in context. It answers questions about the document's subject and leaves other questions alone, which matters when retrieval returns the wrong document. This costs accuracy on unseen phrasings.
- **Limits:**
  - One seed, λ ∈ {0, 1} only, 8 people with city facts, one held-out phrasing per fact.
  - "Unchanged" is first-token only; only 4 capital questions.
  - The general contrast questions include other countries' capitals, so the capital test is close to the contrast distribution. Other people's questions are a cleaner test, and their held-out people and phrasings were never seen.
  - Contrast people share the question templates with the owner, so the intervention may have learned "this name" rather than anything about the facts.

## What we investigate next

1. **Is the KV memory now read depending on the question?** Rerun the attention analysis on `kv_top10_4slots_w1` (the analysis script expects experiment 9's file format and needs a small adapter).
2. **Trade-off curve:** λ between 0 and 1 (e.g. 0.1, 0.3) and several seeds, to compare interventions at matched KL rather than at one λ.
3. **Harder side-effect tests:** general questions far from the contrast set, multi-token answers, and questions that mix the person with general knowledge (e.g. "Which country is the city where X was born in?").
4. **Name-gated memory** (from experiment 9), now compared against contrast training.

## Files

- `outputs/run_contrast.sh`, `outputs/knowledge_contrast.commit` (the code commit the runs used).
- `outputs/<run>/`: `results.csv` (one row per question, with the unmodified model's first token and the KL), `summary.json` and `run.log`.
- `outputs/<run>/interventions.pt` (git-ignored; learned parameters and loss curves). SHA-256:
  - `heads_last_w0`: `25e8ee29a259688ac05f9416d34e5e6c06b76e4db324100a4a264c5c91fe06f1`
  - `heads_last_w1`: `78fcafc91fbec9b78b6bc0aeb09544061154454e482c3b6766ad38ac2865633a`
  - `kv_top10_4slots_w0`: `9d8d952bd4fef990af671df89870acff87a494da8a90403d2631c48dc7e6f8a4`
  - `kv_top10_4slots_w1`: `c31c46bb73be0999457950426b2e24e6691c4a58552cf7dec7e15a1936f6adf2`
  - `layers_head_layers_last_w0`: `c362d221873d81734193661a13e69371f598e0bc82e3c8809c531deb768a14d9`
  - `layers_head_layers_last_w1`: `cd2b2a6cf34fae255c05de02065dba671f8a929d3e8620b3bbf7f143b4695815`
  - `vector_layer12_last_w0`: `3bfe7c06bf56281200949dd410d2f8eef68be8876f0339c93ee707017608b8a1`
  - `vector_layer12_last_w1`: `6004bdef558deb7894b7e547501b3c4edf4a970181413928ddf90e0fe560388f`
  - `vector_layer12_name_w0`: `dc4305d5842d5595a9a2a07e67e760965be6fcb7f6b9b6ac6492929bcc9d3990`
  - `vector_layer12_name_w1`: `5adcdf9040ecd8f5ce95f2e60d31f8e9ffae302c0dc2f61b53750b701a2af4e5`
- `analysis/analyse_contrast.py`: all tables above (column `capitals_wrong_city` counts answers that are one of the 16 dataset cities), held-out accuracy by fact, and what changes on other people's questions.
