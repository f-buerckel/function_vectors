# Balanced and per-fact FVs on Qwen3-4B (2026-11-12)

Follows up the [first injection run](../2026-11-12_qwen3-4b_injection_smoke/README.md), where each person's FV pushed one of their cities for every question. Two changes to how mean activations are averaged test why:

- **Balanced FV:** every fact counts equally. If the single city came from some facts being asked more often during extraction, the FV should stop favouring one city.
- **Per-fact FV:** averaged only over prompts asking about one fact. This tests whether one vector can carry one fact.

Heads are the same as in the [stable extraction run](../2026-11-12_qwen3-4b_extraction_logprob_otherperson/README.md), so only the averaging differs.

## Reproduce

- **Code:** commit `3b4d08cbee1891d37b42323ae22f820b1208b788` on branch `qwen-fv-extraction`. The FVs were built from the identical, then-uncommitted file a minute before the commit.
- **Heads:** indirect effects from the stable extraction run (`fv_extraction.pt`, SHA-256 `85ec83177065d90aab4efb0d0dccbc5a95b965047c6bd331757b02340ffae23f`).
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (setup-only)
- **Commands**, from `src/`:

  ```bash
  python build_fact_fvs.py --extraction_path ../results/knowledge_smoke_logprob_otherperson \
      --save_path ../results/knowledge_smoke_fact_fvs
  python inject_knowledge_fv.py --fv_path ../results/knowledge_smoke_fact_fvs/fact_fvs.pt --fv_key balanced \
      --save_path ../results/knowledge_smoke_injection_balanced
  python inject_knowledge_fv.py --per_fact --fv_path ../results/knowledge_smoke_fact_fvs/fact_fvs.pt \
      --save_path ../results/knowledge_smoke_injection_per_fact
  ```

  Defaults: 10 prompts per fact, 10 heads, layers 0, 3, …, 33 and 35, scales 0.5, 1, 2, 4, 8, `--seed 42`.
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** about 30 s to build the FVs, then about 11 minutes per injection sweep.

## Setup

- **Per-fact means:** for each person and fact, mean activations over 10 prompts. Each prompt has all 4 of the person's facts as demonstrations, and asks a paraphrase of that one fact.
- **Per-fact FV:** built from one fact's mean activations, with the shared top-10 heads.
- **Balanced FV:** built from the average of a person's 4 per-fact means. Because the FV is linear in the mean activations, this equals the average of the 4 per-fact FVs (largest relative difference 0.04%, from float16 rounding).
- **Balanced injection:** the same vectors and metrics as the first injection run, with balanced FVs instead of the original ones.
- **Per-fact injection:** the FV for the fact being asked is chosen in advance, so the question's fact is known. Vectors compared:
  - **own:** the person's FV for the fact asked.
  - **other_fact:** the same person's FV for a different fact.
  - **different_cities_person:** the FV of a person with none of the same cities, for the same fact.
  - **random:** a random vector with the same norm as the own FV.

## Results

**Balanced FVs: no improvement, still one city per person.**

| Own FV, closed-set accuracy | Scale 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Layer 12 | 0.08 | 0.08 | 0.25 | 0.25 | 0.21 |
| Layer 15 | 0.04 | 0.25 | 0.29 | 0.25 | 0.21 |
| Layer 27 | 0.21 | 0.25 | 0.25 | 0.29 | 0.29 |

- From scale 2 upward, accuracy is around 25% at most layers (best 29%), as with the original FVs. All layers are in `outputs/injection_balanced/injection_summary_table.csv`.
- From scale 2 at layers 9 and above, and from scale 4 everywhere, 100% of predictions are one of the person's own cities.
- At scales 4 and 8, each FV gives 1.0–1.33 distinct cities across the 4 questions (averaged over people), so mostly the same city for every question.
- **Balancing changed which city wins, not that one wins.** At layer 15, scale 2, Tomas's FV moved from Vienna to Lima, Elara's from mostly Lisbon to Oslo, and Joren's from Athens to Dublin.
- Twins' balanced FVs are even more alike than before (cosine 0.96–0.98, others 0.36–0.50). The same-cities partner's FV gives the same city as the own FV in 46% of questions at layer 15, scale 2, and 58% at layer 27, scale 8 (29% before balancing).
- The same-cities partner's FV is as accurate as the own FV (25% at layer 27, scale 8), so there is still no binding.

**Per-fact FVs: always the vector's city, whatever the question.**

| Per-fact vector, from scale 2 upward | Right city | Prediction is the city the vector was built from |
|---|---|---|
| own (FV for the fact asked) | 100% (83% at layer 0, scale 2) | 100% |
| other_fact (same person, different fact) | 0% | 100% (79% at layer 0, scale 2) |
| different_cities_person (same fact) | 0–8% | 100% (88% at layer 0, scale 2) |
| random | 0–17% | 0% |

- The question never wins against the vector: at scale 2 and above, the other_fact vector is never right, at any layer.
- At scale 8, the first generated token is the right city in 100% of questions at every layer (own FV). Generations are the city repeated, such as "Madrid Madrid Madrid…".
- At the paper's default (layer 12, scale 1), the own FV gets 71% right, but generations are mostly answer phrasing such as "Answer to the questions about the city of".
- **Per-fact FVs encode the city, not the person or the fact type:**

| Per-fact FV pairs | Mean cosine | Range |
|---|---|---|
| Same city, different person | 0.95 | 0.89–0.98 |
| Same person, different city | 0.56 | 0.33–0.73 |
| Same fact type, different person and city | 0.34 | 0.13–0.70 |
| Nothing shared | 0.33 | 0.10–0.69 |

- FV norms: per-fact 77–127, balanced 82–91. Hidden-state norms at the last token are in the injection summaries.

## Interpretation

- **The averaged FV captures the answer the model was about to give, not knowledge it can query.** Mean activations are taken at the last token, where the model has already worked out the answer to that prompt's question. A per-fact FV is therefore close to a "say this city" vector: FVs from the same city are 0.95 alike across different people, and the model outputs the vector's city whatever is asked.
- **Combining facts doesn't help.** A person's FV is a mix of their answer cities. The model then outputs whichever city dominates the mix, not the one the question asks for. Balancing the mix changes which city dominates, but one still wins.
- **This is the "pushes one answer token" failure** from the project plan: a correct answer alone doesn't show stored knowledge. Per-fact FVs reach 100% only because the right vector is chosen for each question, which amounts to looking up the answer.
- **So, for this FV construction:** one vector can carry one answer, but a person's facts can't be stored in one vector so that the question picks the right one.
- **Limits:**
  - Only 6 people and 24 questions.
  - Only last-token extraction and injection, with the paper's head selection.
  - Best configurations are chosen on the reported questions.

## What we investigate next

1. **Extract from somewhere other than the last token.** The last token already holds the answer to one specific question. Activations at the person's name in a document or question may hold facts about the person without committing to one answer.
2. **Inject at the person's name** in the question, so the model's own attention can pick the fact the question asks for.
3. **Learned vector:** optimise one vector per person to answer all 4 questions, and test it on held-out paraphrases. This shows whether one vector at one position can hold four bound facts at all, independently of how FVs are extracted.

## Files

- `outputs/fact_fvs/`: `summary.json` (heads, similarity tables, norms). `fact_fvs.pt` (163 kB) stays local because `*.pt` is git-ignored; its SHA-256 is `877966530ceec17430655bbe12d32e6aba375984f8db665fdabac12b4ce61a35`.
- `outputs/injection_balanced/` and `outputs/injection_per_fact/`: `injection_results.csv`, `injection_summary_table.csv`, `summary.json` and `run.log`.
- `analysis/analyse_balanced_and_per_fact.py`: the tables above. It needs the first injection run's outputs for the unbalanced comparison.
