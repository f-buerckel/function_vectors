# Learned knowledge vectors on Qwen3-4B (2026-11-12)

After [balanced and per-fact FVs](../2026-11-12_qwen3-4b_balanced_and_per_fact_fvs/README.md) showed that extracted FVs push a single answer, this tests whether a single vector can hold a person's facts **at all**. One vector per person is optimised by gradient descent so the frozen model answers all 4 of that person's questions. This is an upper bound: if an optimised vector can't hold the facts at a layer and position, no extraction method that produces a vector there can.

## Reproduce

- **Code:** commit `56a3dad4c35c8aa68738e6c6677788c4d354519c` on branch `qwen-fv-extraction`
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (setup-only)
- **Command**, from `src/`:

  ```bash
  python learn_knowledge_vector.py --save_path ../results/knowledge_smoke_learned_vector
  ```

  Defaults: positions `last` and `name`, layers 6, 12, 18, 24, 150 Adam steps with learning rate 0.1, the vector starting at zero, loss scale 1024 (against float16 gradient underflow), `--seed 42`.
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** about 11 minutes for 48 vectors (6 people × 2 positions × 4 layers).

## Setup

- **Training questions:** for each fact, the demonstration question and both extraction paraphrases, so 12 per person, asked bare (`Q: …\nA:`).
- **Loss:** cross-entropy of the answer city's token at the last prompt token, over the full vocabulary.
- **Positions where the vector is added:**
  - `last`: the last prompt token, where FVs are added.
  - `name`: the last token of the person's name in the question.
- **Tests:**
  - **Held-out paraphrases:** the 4 test questions per person, never used in training.
  - **Twin's questions:** the vector applied to the held-out questions of the person with the same cities in other roles.
  - **Known facts:** 4 capital questions (France, Germany, Italy, Japan), to check for side effects. These questions contain no person's name, so `name` vectors are added at their last token instead.
- **Baseline:** a bag of answers that always gives one of the person's cities gets 25%, which is what FVs reached.

## Results

**One vector holds all four bound facts.** Every vector gets all 12 training questions right (final loss at most 0.007).

| Position | Layer | Held-out right (closed set and first token) | People with all 4 held-out right | Known facts still right | Vector norm |
|---|---|---|---|---|---|
| last | 6 | **0.88** | 4 of 6 | 0.50 | 37–44 |
| last | 12 | 0.67 | 1 of 6 | 0.75 | 37–42 |
| last | 18 | 0.75 | 0 of 6 | 0.25 | 47–51 |
| last | 24 | **0.88** | 3 of 6 | 0.17 | 71–83 |
| name | 6 | 0.71 | 1 of 6 | 0.96 | 33–39 |
| name | 12 | 0.71 | 2 of 6 | **1.00** | 40–47 |
| name | 18 | 0.75 | 2 of 6 | 0.83 | 55–61 |
| name | 24 | 0.79 | 3 of 6 | 0.33 | 108–124 |

- Without a vector, 0 of 24 held-out questions are right.
- On held-out questions, the closed-set and full-vocabulary first-token accuracies are identical: the model's first token is the city.
- Held-out accuracy is 67–88% everywhere, far above the 25% of a bag of answers.

**Errors are binding errors, not missing facts.** All 45 wrong held-out answers (over all positions and layers) are another of the person's own cities. Most come from a few questions:

| Held-out question | Wrong in (of 8 configurations) |
|---|---|
| Which city is the birthplace of Davor Quill? | 6 |
| Which city is the birthplace of Joren Thale? | 6 |
| Which city does Mira Solbrandt live in? | 5 |
| Which city is the birthplace of Tomas Wrenfield? | 4 |

The study-city paraphrase is right in every configuration; the "birthplace" paraphrase is the hardest.

**Last-token vectors ignore the name; name vectors are more selective.** On the twin's held-out questions:

| Position | Owner's answer for that question type | Twin's answer | Something else |
|---|---|---|---|
| last (layers 6–24) | 0.58–0.79 | 0.00–0.12 | 0.17–0.29 |
| name (layers 6–24) | 0.17–0.42 | 0.21–0.25 | 0.38–0.62 |

A last-token vector answers "born in?" with its owner's birth city whoever is asked about. Added at another person's name, a name vector has much less of this effect.

**Side effects:** last-token vectors damage unrelated questions (17–75% of capitals still right), while name vectors at layers 6 and 12 leave them almost untouched (96–100%). Since capital questions have no name, `name` vectors are added at their last token there, so this compares the vectors' strength rather than their position.

## Interpretation

- **Capacity is not the problem.** A single vector of the model's hidden size, added at one position, can hold a person's 4 facts bound to their question types, and mostly generalises to new phrasings. The FVs' failure to bind comes from how they are extracted, not from the single-vector format.
- **Last-token vectors behave like a lookup table keyed by question type:** they answer from the vector whoever is asked about. This is consistent with the FV results: at the last token, a vector overrides the question rather than working with the model's own retrieval.
- **Name-position vectors fit how the model retrieves facts:** they are tied less to the question type alone, and at early-middle layers they hardly affect other questions. This makes the name position the more promising place for extracted vectors.
- **Limits:**
  - Vectors are trained per person on 12 questions about that person, so this is per-document training (closer to PRAG than to FVs).
  - 6 people and 4 held-out questions each.
  - Generalisation was only tested on one held-out paraphrase per fact.

## What we investigate next

1. **Extract a vector at the person's name without training:** the mean difference between the name token's hidden state with and without the facts in context. If it approaches the learned name vectors, document knowledge can be extracted without per-document optimisation. Started as the context-difference experiment.
2. **Learn with more paraphrases,** to improve generalisation to unseen phrasings like "birthplace".
3. **Amortise the learned vector:** train a small network that maps a document to its vector (as DyPRAG does for LoRA adapters).

## Files

- `outputs/learned_vector_results.csv`: one row per question, vector (learned or none), position and layer.
- `outputs/summary.json`: accuracy tables, known-fact results, final losses and vector norms.
- `outputs/run.log`
- `outputs/learned_vectors.pt` (572 kB, git-ignored): the learned vectors and loss curves. SHA-256 `a8b158f597d9b570251e274bfa34af87dbfda57d3244c38b58d2da7bddfaeb65`.
- `analysis/analyse_learned_vector.py`: errors by question type and twin behaviour.
