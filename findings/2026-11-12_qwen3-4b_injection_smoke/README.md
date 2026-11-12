# FV injection into bare questions on Qwen3-4B (2026-11-12)

The first test of whether an extracted function vector (FV) makes the model answer. Each person's FV, from the [log-probability, other-person extraction run](../2026-11-12_qwen3-4b_extraction_logprob_otherperson/README.md), is added to bare questions about that person, with no demonstrations.

## Reproduce

- **Code:** commit `95c9eb684ca6029035acb9fe24e807a6b044ee64` on branch `qwen-fv-extraction`
- **FVs:** `fv_extraction.pt` from the extraction run (SHA-256 `85ec83177065d90aab4efb0d0dccbc5a95b965047c6bd331757b02340ffae23f`)
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (setup-only)
- **Command**, from `src/`:

  ```bash
  python inject_knowledge_fv.py --fv_path ../results/knowledge_smoke_logprob_otherperson/fv_extraction.pt \
      --save_path ../results/knowledge_smoke_injection
  ```

  Defaults: layers 0, 3, …, 33 and 35; scales 0.5, 1, 2, 4, 8; `--seed 42`.
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** about 11 minutes for 6,240 interventions.

## Setup

- **Questions:** the 24 held-out paraphrases (4 per person), asked bare: `Q: Which city is the birthplace of Elara Voss?\nA:`
- **Intervention:** the vector, times a scale, is added to the hidden state at the last token of one layer.
- **Vectors compared:**
  - **own:** the FV of the person asked about.
  - **same_cities_partner:** the FV of the person with the same 4 cities in other roles.
  - **different_cities_person:** the FV of a person sharing none of the cities.
  - **random:** a random vector with the same norm as the own FV.
- **Main metric:** which of the 12 dataset cities scores highest (closed set, chance 1/12). Also recorded:
  - whether that city is one of the vector owner's 4 cities;
  - whether it is the owner's answer for the question type asked;
  - the change in the answer's log-probability;
  - the top token over the full vocabulary.

## Results

**Without a vector, the model doesn't know the facts.** 0 of 24 correct, with Tokyo and Oslo as the most common guesses.

**With the own FV, accuracy settles at exactly 25%.** Closed-set accuracy of the own FV:

| Layer | Scale 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 0 | 0.08 | 0.12 | 0.17 | 0.25 | 0.25 |
| 12 (paper's default) | 0.04 | 0.12 | 0.25 | 0.25 | 0.25 |
| 15 | 0.04 | 0.21 | 0.29 | 0.25 | 0.25 |
| 24 | 0.12 | 0.21 | 0.25 | 0.25 | 0.25 |
| 27 | 0.21 | 0.25 | 0.25 | 0.25 | 0.25 |
| 35 | 0.08 | 0.12 | 0.25 | 0.25 | 0.25 |

All 13 layers are in `outputs/injection_summary_table.csv`. The best value, 0.29 at layer 15 and scale 2, is chosen on the same questions it is reported on, so it is optimistic.

**The FV pushes one of the person's cities, the same one for every question.**
- From scale 2 at layers 9 and above, and from scale 4 everywhere, 100% of predictions are one of the person's own 4 cities.
- At scale 8, and at scale 4 on every layer except 21, each person's FV gives the same city for all 4 questions. With 4 questions per person, that makes exactly 1 of 4 correct, hence 25%.
- At layer 15, scale 2, the FV pushes Lisbon for Davor, Dublin for Ines, Athens for Joren and Vienna for Tomas, whatever the question. Elara gets Lisbon three times and Oslo once, and Mira gets Lima three times and Madrid once.

| Vector at layer 15, scale 2 | Correct | Prediction is the owner's answer for that question type | Prediction is one of the owner's cities | Change in answer log-probability |
|---|---|---|---|---|
| own | 0.29 | 0.29 | 1.00 | +5.1 |
| same_cities_partner | 0.25 | 0.25 | 1.00 | +4.9 |
| different_cities_person | 0.00 | 0.25 | 1.00 | −2.4 |
| random | 0.04 | 0.00 | 0.00 | −3.8 |

- **There is content, not just the task:** another person's FV makes the model answer with that person's cities, and a random vector doesn't.
- **There is no binding:** the own FV is no better than the same-cities partner's FV, and neither is above 25%.

**Which city is pushed seems to follow the extraction prompts.** Mean activations average 10 clean prompts whose questions are drawn at random. Replaying those draws (`analysis/replay_extraction_queries.py`) shows that in 4 of 6 people, the pushed city belongs to the fact asked most often:

| Person | Questions in the 10 extraction prompts | Fact whose city the FV pushes |
|---|---|---|
| Tomas | study 7, the others 1 each | study |
| Ines | born 5, study 3 | born |
| Mira | study 5, work 3, born 2 | study (work once) |
| Joren | home 4, work 3 | home |
| Elara | born 3, study 3, 2 each for the others | born (work once) |
| Davor | home 3, study 3, 2 each for the others | work |

This also explains why same-city twins can push different cities despite FV cosine similarity of 0.86–0.97: their extraction prompts asked about different facts. The replay mirrors the code's random calls but was not checked against the actual prompts.

**The first token is never the city.** Full-vocabulary top-1 accuracy is 0 for the own FV. The model starts with "Answer" or "The answer is", and cities appear later in generations:
- At layer 15, scale 2: "Answer is is Lisbon Lisbon Lisbon…" for Elara, and "Answer: Lisbon, Oslo, Lisbon," for Davor.
- At layer 12, scale 1: only answer phrasing, such as "The answer is: I don't know".

These generations add the FV at every generated token.

**The FV is large compared with the hidden state at early layers.** The hidden state at the last token has a norm of about 38 at layer 12 and 41 at layer 15, against 650 at layer 35. FV norms are 76–94, so at layer 15 and scale 2 the FV is about 4 times the hidden state and dominates it. At layer 27 (norm about 150), scale 1 is enough to reach 25%.

## Interpretation

- **The FV carries content:** it makes the model output the person's cities instead of guessing, and another person's FV gives that person's cities. This is more than a task signal.
- **It doesn't store which answer belongs to which question:** the FV acts like a weighted mix of the answers seen during extraction and pushes the strongest one, whatever is asked. The extraction-run hypothesis of a "bag of answers" is confirmed, with the twist that the bag is weighted by which questions happened to be drawn.
- **Format comes with the content:** the FV also carries the "answer the question" format, which shows up as "The answer is" before any city.
- **Limits:** this is a 6-person setup dataset with 4 questions each, the best configuration is not held out, and only first tokens and short generations were checked.

## What we investigate next

1. **Balanced extraction prompts:** ask every fact equally often when computing mean activations. If the FV then pushes no single city, or a mix, that confirms the weighting explanation.
2. **One FV per fact:** average only over prompts asking about one fact. If that FV gives the right city, a vector can carry one fact, and the problem is combining several facts in one vector.
3. **Other injection positions:** add the FV at the person's name in the question rather than the last token, so the model's own attention can select the right fact.
4. **Learned vector:** optimise a vector directly to answer all 4 questions, to see whether one vector at one layer can hold four bound facts at all.

## Files

- `outputs/injection_results.csv`: one row per intervention.
- `outputs/injection_summary_table.csv`: metrics by vector, layer and scale.
- `outputs/summary.json`: clean baseline, both configurations, generations, hidden-state and FV norms.
- `outputs/run.log`
- `analysis/analyse_injection.py`: per-question predictions and the numbers above. Run it on `gpu-pc` or anywhere with pandas.
- `analysis/replay_extraction_queries.py`: reconstruction of which questions the extraction prompts asked (numpy only).
