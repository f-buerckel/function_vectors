# Learned key–value memory for attention heads on Qwen3-4B (2026-11-12)

**Branch `kv-memory`.** The [per-head and per-layer vectors](../2026-11-12_qwen3-4b_learned_heads_and_layers/README.md) were accurate on held-out questions but answered every question with the person's cities, because a fixed vector acts the same way on every prompt. This tests the variant meant to fix that: instead of a fixed output, each selected head gets **learned key–value entries**, as if a few extra tokens were in its context. The head's own query, which depends on the question, decides how much to read from them, like in-context retrieval from a document.

## Reproduce

- **Code:** commit `d95b56924b89f386177e7b5435cb7427d3768ac8` on branch `kv-memory` (runs). Commit `5632189` adds attention recording (off by default) and the attention analysis.
- **Data:** `dataset_files/knowledge/fictional_people_capacity.json` (8 people × 8 city facts, setup-only), all 8 facts.
- **Commands:** `outputs/run_kv.sh`. Each run is `learn_kv_memory.py --n_facts 8` with 300 Adam steps, learning rate 0.05 and loss scale 1024, plus:

| Run | Options |
|---|---|
| `top10_heads_4_slots` | `--n_slots 4` |
| `top10_heads_16_slots` | `--n_slots 16` |
| `layers_6_12_18_24_all_heads_2_slots` | `--layers_all_heads 6 12 18 24 --n_slots 2` |

  The attention analysis, from `src/`: `python ../findings/2026-11-12_qwen3-4b_kv_memory/analysis/memory_attention.py ../findings/2026-11-12_qwen3-4b_kv_memory/outputs/<run>`.
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16 model with float32 attention in memory layers.
- **Runtime:** about 22 minutes for the 3 runs.

## Setup

- **Heads:** the same top-10 FV heads as the per-head vectors, or every head of layers 6, 12, 18 and 24.
- **Memory per head:** m learned keys and values (128 numbers each) and one learned logit bias. The head's query attends over the real tokens and its memory entries in one softmax.
- **Starting point:** values start at zero and the bias at −4, so the memory starts nearly inactive.
- **Visibility:** memory entries are visible from every position, like a prefix.
- **Implementation:** transformers' attention function is wrapped. Without memory, logits are identical to the unmodified model. With the memory blocked (bias −10,000), the float32 path differs by at most 0.016 on logits up to 22, which is float16 rounding.
- **Tests:** as before, held-out paraphrases and 4 capital questions. The attention analysis also uses each person's memory on the next person's held-out questions.

## Results

| Intervention (8 facts) | Numbers per person | Held-out right | People with all 8 right | Capitals still right |
|---|---|---|---|---|
| Single learned vector (experiment 7) | 2,560 | 0.58–0.77 | 0/8 | 0.66–0.81 |
| 10 head vectors, tuned (experiment 8) | 1,280 | 0.92 | 3/8 | 0.00 |
| Memory, top-10 heads, 4 entries | 10,250 | 0.88 | 2/8 | 0.00 |
| Memory, top-10 heads, 16 entries | 40,970 | 0.91 | 3/8 | 0.00 |
| Memory, all heads of layers 6/12/18/24, 2 entries | 65,664 | 0.52 | 0/8 | 0.75 |

- Training questions are 100% right in every run (largest final loss 0.002).
- All held-out errors are another of the person's own cities (8, 6 and 31 errors respectively).

**The memory is not read selectively.** Mean share of the selected heads' attention on the memory, at the last token:

| Questions asked | Top-10 heads, 4 entries: attention on memory | Answer right | Answer is one of the memory owner's cities | Early layers, 2 entries: attention on memory | Answer right | Answer is one of the memory owner's cities |
|---|---|---|---|---|---|---|
| The person's own held-out questions | 0.79 | 0.88 | 1.00 | 0.08 | 0.52 | 0.98 |
| Another person's held-out questions | 0.79 | 0.08 | 1.00 | 0.08 | 0.06 | 0.95 |
| Capital questions | 0.83 | 0.00 | 0.91 | 0.07 | 0.75 | 0.19 |

- At the top-10 heads, the memory takes about 80% of the heads' attention whatever is asked (63–87% per question). The heads have learned to read mostly from the memory, not to look things up depending on the question.
- As a result, a person's memory makes the model answer another person's questions with the memory owner's cities (100%) and capitals with them too (91%).
- In early layers, the memory takes only 5–12% of attention and leaves capitals mostly intact, but it still answers other people's questions with the owner's cities (95%), and generalises worse (52%).

## Interpretation

- **Question-dependent retrieval doesn't emerge on its own.** The mechanism allows it, but every training question was about the memory's owner, so reading the memory on every prompt is the best solution to the training objective. The same holds for the fixed vectors, so the side effects come from the training data, not the parameterisation.
- **Key–value memory at the FV heads is about as accurate as fixed head vectors** (88–91% against 92%) and has the same side effects, while needing 8–32 times more numbers.
- **The next step applies to all learned variants:** train with contrast questions, such as other people's questions and general questions, where the model must keep its normal answer (for example, a loss to match the unmodified model's predictions). Only then does held-out accuracy measure knowledge rather than an override, and only then can the key–value memory show whether question-dependent reading pays off.
- **Limits:**
  - 8 people, cities only, one held-out phrasing per fact.
  - Only 4 capital questions.
  - Memory entries are visible from every position.
  - Fixed training settings.

## What we investigate next

1. **Contrast training:** add other people's questions and general questions with a "keep the original answer" loss, for vectors, head vectors and key–value memory alike, and compare them at equal side effects.
2. **Name-gated memory:** make memory entries visible only to tokens after the person's name, as a simple form of document-specific retrieval.

## Files

- `outputs/<run>/`: `results.csv`, `summary.json` and `run.log` for each run, plus `outputs/runner.log` and `outputs/run_kv.sh`.
- `outputs/<run>/memory_attention.csv`: the attention analysis (for the top-10, 4-entry run and the early-layer run).
- `outputs/<run>/memories.pt` (git-ignored). SHA-256:
  - `top10_heads_4_slots`: `34dcce4054d988742152e57d304190d70165f0603044a3780028f2835e2af316`
  - `top10_heads_16_slots`: `b1808451a61869fbb24469a3a769d0ebae92ba2959dcf8dfd3ca3d60bec7abb4`
  - `layers_6_12_18_24_all_heads_2_slots`: `cf96eeea97f055ad65fe1712ee4b3ba3848945803f75a3481409faad31cc0407`
- `analysis/memory_attention.py`: attention on the memory and answers by question type (needs the GPU).
