# Context-difference vectors at the name on Qwen3-4B (2026-11-12)

The [learned vectors](../2026-11-12_qwen3-4b_learned_vector/README.md) showed that one vector at the person's name can hold their bound facts, and barely affects other questions. This tests whether such a vector can be **extracted** without training: the difference the facts in context make to the hidden state at the person's name.

## Reproduce

- **Code:** commit `56af01226d0c4e33e1dca28c84b7461cd2889073` on branch `qwen-fv-extraction`
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (setup-only)
- **Command**, from `src/`:

  ```bash
  python context_diff_vector.py --save_path ../results/knowledge_smoke_context_diff
  ```

  Defaults: positions `name` and `last`, layers 3, 6, 9, 12, 18, 24, scales 1, 2, 4 (plus 0 as a no-vector baseline), 2 demonstration orders per extraction question, `--seed 42`.
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** about 2 minutes.

## Setup

- **Extraction**, for each person and each of their 8 extraction questions, with 2 random orders of the demonstrations:
  - **Clean prompt:** all 4 facts as demonstrations, then the question.
  - **Bare prompt:** the question alone.
  - The hidden state after each layer is taken at the same token in both, at one of two positions:
    - `name`: the last token of the person's name in the question. In the clean prompt this token can attend to all the facts.
    - `last`: the last prompt token.
  - The vector is the mean difference (clean minus bare) over the 16 prompts. No head selection and no training.
- **Injection:** the vector, times a scale, is added at the same position and layer of the bare held-out questions.
- **Tests:** as for the learned vectors: held-out paraphrases, the twin's questions, and 4 capital questions (where `name` vectors go to the last token, as these questions have no name).

## Results

**The extracted vector doesn't make the model answer.**

| Position | Layers | Held-out right (closed set) | Prediction is one of the person's cities |
|---|---|---|---|
| name | 3–24 | 0% at every layer and scale | 8–21% (no vector: 12%) |
| last | 3–18 | 0–4% | 0–17% |
| last | 24 | 21% (scale 1), 29% (scale 2), 17% (scale 4) | 58%, 96%, 67% |

- Only the last token at layer 24 has an effect, and it is the bag-of-answers pattern seen with FVs: one of the person's cities, not the one asked for.
- Larger scales damage unrelated questions: at scale 4, capitals are only 0–33% right for `name` vectors (added at the last token there) and 21% for `last` vectors at layer 24.

**Why: the difference is mostly generic, and its person-specific part points elsewhere.**

| Position | Layer | Norm | Cosine between people | Share of squared norm common to all people | Cosine with learned vector, person-specific part |
|---|---|---|---|---|---|
| name | 6 | 13.2 | 0.49 | 0.56 | +0.01 |
| name | 12 | 26.1 | 0.56 | 0.63 | +0.03 |
| name | 18 | 38.5 | 0.69 | 0.74 | +0.03 |
| name | 24 | 90.2 | 0.48 | 0.50 | +0.11 |
| last | 6 | 14.1 | 0.99 | 0.99 | +0.00 |
| last | 12 | 27.5 | 0.99 | 0.99 | +0.00 |
| last | 18 | 28.6 | 0.98 | 0.98 | +0.01 |
| last | 24 | 58.1 | 0.94 | 0.95 | +0.09 |

The person-specific part is the vector minus the mean over all 6 people.

- At the last token, 95–99% of the difference is shared by all people: a generic "there was context" signal, with almost nothing about the person.
- At the name, about half is person-specific, so the context does change the model's representation of the person. But that part is nearly orthogonal to the learned name vector that works.

## Interpretation

- **The facts in context don't end up in the name token in a form the model can use later.** A plausible reason (not tested here): with the facts in context, the model answers by attending from the answer position to the demonstrations, rather than by first writing the facts into the representation of the person. That would explain why neither the name nor the last token carries extractable facts.
- **Learned and extracted vectors are unrelated.** Optimisation finds a direction that works, but the model's own processing of the document doesn't produce it. Many different vectors can cause the same behaviour, so low cosine alone doesn't prove the extracted vector lacks the facts, but together with 0% accuracy it does.
- **So far, only optimisation produces a working single-vector representation** of a person's facts. Simple extraction, whether head-based FVs or context differences, doesn't.
- **Limits:** one simple extraction method, 6 people, and only the last name token.

## What we investigate next

1. **Capacity of learned vectors:** how many facts one vector can hold before held-out accuracy drops. This is what a comparison with PRAG's per-document LoRA needs.
2. **Amortised vectors:** train a network that maps a document to its learned vector (as DyPRAG does for LoRA), since direct extraction fails.
3. **Test the retrieval explanation:** measure where attention in clean prompts goes when the answer is produced (to the demonstrations' answer tokens, or the name).

## Files

- `outputs/context_diff_results.csv`: one row per question, position, layer and scale.
- `outputs/summary.json`: accuracy tables, known-fact results and vector norms by layer.
- `outputs/run.log`
- `outputs/context_diff_vectors.pt` (4.4 MB, git-ignored): the extracted vectors for all layers. SHA-256 `9e5c616924d7e09da73586258ca216a8cba70d82d838e8a0ca6d87ca9e655bf7`.
- `analysis/compare_with_learned.py`: the similarity table. It needs the learned-vector outputs (`learned_vectors.pt`).
