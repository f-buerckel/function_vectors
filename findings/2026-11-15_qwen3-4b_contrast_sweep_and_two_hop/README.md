# Contrast-weight sweep and two-hop evaluation on Qwen3-4B (2026-11-15/16)

**Branch `contrast-training`.** Follow-up to [contrast training](../2026-11-15_qwen3-4b_contrast_training/README.md) (experiment 10), with two parts:

1. **λ sweep:** the contrast weight at 0.1, 0.3 and 3, in addition to experiment 10's 0 and 1, for head vectors, the vector at the name and the key–value memory. Plus a second seed at λ = 1 for head vectors and the name vector.
2. **Generation-based evaluation** of all 21 trained runs (experiment 10 and the sweep). Experiment 10 scored only the first token. This evaluation generates full greedy answers and adds:
   - **two-hop questions** ("In which country was {name} born?"), which test whether the stored fact can be used in reasoning;
   - general questions unlike the contrast set;
   - the key–value memory's attention share.

## Reproduce

- **Sweep code:** commit `5344331f06078aa0db2b2c197ffd336465de3148` (`src/learn_knowledge_contrast.py`, unchanged from experiment 10). Commands: `outputs/sweep/run_sweep.sh`; 300 steps each, other options as in experiment 10.
- **Evaluation code:** commit `5a253807c018e39541a1715ab3db0fdfea52dc7a`, `src/eval_contrast_interventions.py`. Command: `outputs/generation_eval/run_eval.sh`, run from `src/` on the 10 run folders of experiment 10 and the 11 sweep folders. Greedy decoding with 16 new tokens, each answer cut at its first line. The intervention stays at the prompt's position (last token or name) while generating.
- **Data:** `dataset_files/knowledge/fictional_people_capacity.json` (8 people × 8 city facts, setup-only).
- **Analysis**, from the repo root: `python findings/2026-11-15_qwen3-4b_contrast_sweep_and_two_hop/analysis/analyse_sweep_and_two_hop.py findings/2026-11-15_qwen3-4b_contrast_sweep_and_two_hop/outputs findings/2026-11-15_qwen3-4b_contrast_training/outputs` (pandas only).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16 model (float32 attention in memory layers).
- **Runtime and interruptions:**
  - The sweep took 3 h 2 min (2026-11-15, 13:11–16:13).
  - The evaluation took 23 minutes (2026-11-16, 19:47–20:11).
  - A first evaluation attempt on 2026-11-15 was stopped deliberately before it finished its first run and wrote nothing. Before that, a CPU test on one person checked the script with 10 new tokens; it's not part of these results.

## Setup of the evaluation

Sets per person, all answered with the person's intervention active:

| Set | Questions | Scored as right if the answer contains |
|---|---|---|
| Own held-out | 8 held-out paraphrases | the city |
| Own two-hop | 8, e.g. "In which country was Arlen Tovik born?" | the city's country (US: "United States", "USA", "U.S.") |
| Other people, held-out and two-hop | 24 + 24, about the 3 people not used as contrast | (checked for the owner's cities or countries) |
| Capitals | the 4 capital questions | the capital |
| General, city answers | 6 (Golden Gate Bridge, Taj Mahal, Venice, Big Ben, Kremlin, Hollywood) | the city |
| General, other answers | 14 (Mona Lisa, Red Planet, …) | the reference answer |

- **References:** every set is also answered by the unmodified model. The person's own questions are also answered with the facts in context, using the person's 8 demonstration Q/A pairs as in extraction.
- **Two-hop bag-of-answers baseline** (the country of a random one of the person's cities): 12.5%.

## Results

### 1. λ sweep (first-token metrics, as in experiment 10)

| Intervention | λ | Held-out right | Capitals right | Other people → owner's city | KL, other people |
|---|---|---|---|---|---|
| 10 head vectors | 0 | 0.92 | 0.00 | 1.00 | 16.6 |
| | 0.1 | 0.86 | 0.44 | 0.07 | 0.86 |
| | 0.3 | 0.84 | 0.62 | 0.02 | 0.61 |
| | 1 | 0.86 | 0.78 | 0.01 | 0.37 |
| | 1 (seed 43) | 0.83 | 0.66 | 0.01 | 0.37 |
| | 3 | 0.78 | 0.75 | 0.01 | 0.30 |
| Key–value memory | 0 | 0.88 | 0.00 | 0.99 | 12.1 |
| | 0.1 | 0.84 | 0.88 | 0.08 | 0.71 |
| | 0.3 | 0.78 | 1.00 | 0.04 | 0.36 |
| | 1 | 0.75 | 0.88 | 0.01 | 0.15 |
| | 3 | 0.70 | 0.97 | 0.01 | 0.07 |
| Single vector at the name | 0 | 0.66 | 0.75 | 0.45 | 7.9 |
| | 0.1 | 0.55 | 0.84 | 0.00 | 0.11 |
| | 0.3 | 0.58 | 0.88 | 0.00 | 0.05 |
| | 1 | 0.59 | 0.88 | 0.00 | 0.03 |
| | 1 (seed 43) | 0.55 | 0.91 | 0.00 | 0.04 |
| | 3 | 0.53 | 0.84 | 0.00 | 0.02 |

- Even λ = 0.1 removes most of the override. Larger λ trades held-out accuracy for smaller side effects.
- The second seed moves held-out accuracy by 3–5 points.

### 2. Full answers and two-hop questions at λ = 1 (seed 42)

| Intervention | Numbers per person | Held-out right (full answer) | Two-hop right | Capitals right | General, city answers | General, other answers |
|---|---|---|---|---|---|---|
| The heads' 8 layers | 20,480 | 0.83 | **0.81** | 1.00 | 0.83 | 1.00 |
| Key–value memory | 10,250 | 0.75 | **0.70** | 1.00 | 0.85 | 1.00 |
| 10 head vectors | 1,280 | 0.83 | 0.44 | 1.00 | 0.88 | 0.99 |
| Single vector, name | 2,560 | 0.56 | 0.33 | 1.00 | 0.88 | 1.00 |
| Single vector, last token | 2,560 | 0.47 | 0.31 | 1.00 | 0.90 | 1.00 |
| *Unmodified model* | | 0.00 | 0.02 | 1.00 | 0.83 | 1.00 |
| *Facts in context* | | | 1.00 | | | |

- **Right two-hop answers recall the city, then add the country,** for example "Sydney, Australia" and "Chicago, Illinois, United States".
- **Wrong two-hop answers** name another of the person's countries in 36–77% of cases. Some answers name only a city ("Madrid", "Prague"), which counts as wrong even when it's the right city.
- **Capitals are 100% right with full answers for every λ ≥ 0.1 run.** First-token scoring had shown 69–88% at λ = 1 because answers began with "The".
- **General questions with city answers** are 79–92% right across contrast runs, around the unmodified model's 83%. The unmodified model's miss is the Hollywood question, where it doesn't name Los Angeles.
- **Other people's two-hop questions** name the owner's countries 25–36% of the time at λ = 1. The unmodified model does so 29% of the time by chance (it often says "United States"). At λ = 0, head vectors and the 8 layers do so 85% and 81% of the time.

### 3. Two-hop accuracy across λ

| Intervention | λ = 0 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|
| 10 head vectors | 0.91 | 0.80 | 0.69 | 0.44 (seed 43: 0.47) | 0.28 |
| Key–value memory | 0.23 | 0.61 | 0.73 | 0.70 | 0.64 |
| Single vector at the name | 0.34 | 0.31 | 0.31 | 0.33 (seed 43: 0.33) | 0.28 |

- Over the same range, held-out accuracy (full answer) of the head vectors moves only from 0.92 to 0.77.

### 4. Key–value memory: attention share on the memory at the prompt's last token

| λ | Owner's held-out questions | Other people's held-out questions | General questions (capitals and general) |
|---|---|---|---|
| 0 | 0.79 | 0.79 | 0.80 |
| 0.1 | 0.76 | 0.74 | 0.60 |
| 0.3 | 0.73 | 0.68 | 0.42 |
| 1 | 0.67 | 0.55 | 0.28 |
| 3 | 0.55 | 0.28 | 0.12 |

## Interpretation

- **Recall and use come apart.**
  - On direct held-out questions, the 10 head vectors are the most accurate per number (experiment 10).
  - On two-hop questions with contrast, they fall to 44%. The vectors in the heads' 8 layers (81%) and the key–value memory (70%) let the model use the stored fact almost as often as they let it recall it.
  - So held-out accuracy alone overrates the head vectors as a knowledge store.
- **Contrast affects the variants in opposite directions on two-hop questions.**
  - Head vectors lose the ability as λ grows (91% → 28%), while their direct accuracy barely moves.
  - The key–value memory gains it (23% → 70%), because without contrast its override breaks the answer.
  - A hypothesis, not tested: with contrast, head vectors learn to act only on the trained "which city" question form, and the "which country" form falls outside it. The memory instead learns when to read.
- **The key–value memory now reads depending on the question.** With contrast, the attention share separates by question type: highest for the owner, lower for other people, lowest for general questions (λ = 1: 67 / 55 / 28%). At λ = 3 the separation is clearest (55 / 28 / 12%). Experiment 9 found no such dependence without contrast, and this run reproduces that (79% for everything).
- **Side effects are small with full answers.** The lower capital scores in experiment 10 came from first-token scoring, not from wrong answers.
- **The single vector at the name** has the smallest side effects, but its knowledge is among the least usable (28–34% two-hop at every λ).
- **Limits:**
  - One seed for most settings, 8 fictional people, city facts only.
  - Two-hop questions have one template per fact type and only test city → country.
  - Answers are scored by substring, so a bare city counts as wrong and a list of several countries would count as right; the inspected examples show no such lists.
  - Only 20 general questions.

## What we investigate next

1. **Why head vectors lose two-hop ability with contrast:**
   - check whether it's the question form, using "which city" two-hop variants and country questions in training;
   - try head vectors at more heads.
2. **Comparison with PRAG** (a LoRA per document) at matched parameters, on the same tests including two-hop questions.
3. **Real documents:** multi-token answers, longer texts, questions that combine several facts.
4. **Amortisation:** predict the key–value memory or layer vectors from the document in one pass, removing the per-document training.

## Files

- `outputs/sweep/<run>/`: `results.csv`, `summary.json` and `run.log` per sweep run, plus `run_sweep.sh` and `knowledge_contrast_sweep.commit`.
- `outputs/generation_eval/knowledge_contrast*/<run>/generation_eval.csv` (one row per question with the modified, unmodified and in-context answers) and `generation_eval_summary.json`. These cover all 21 runs; the experiment-10 runs' training outputs are in that experiment's folder. Also `run_eval.sh`, `run_eval.log` and `generation_eval.commit`.
- `outputs/sweep/<run>/interventions.pt` (git-ignored). SHA-256:
  - `heads_last_w0.1`: `c98441d4254156476702ff316595eb59c89c65875b66ba5de4f5661c9e12c2f6`
  - `heads_last_w0.3`: `d17e30b29f1a3cfcb33665b414e6bbc8b5b841e1f67840c53efbf7b8fcdf5df9`
  - `heads_last_w1_seed43`: `3fb0ef49d7614ca9a48506cab47c8287ba0db5e96a8f1973612cfff83421f9b7`
  - `heads_last_w3`: `d84402a5dcc5ce4eab22d8a5cac9d7aad8eaf79c933cfddb7405e399e7cba152`
  - `kv_top10_4slots_w0.1`: `b42590da6662c250d6e4ee6dbf27ee4d689069c66604478e2e0ad74d7964536d`
  - `kv_top10_4slots_w0.3`: `43e172323f0c9cb84651ea1b85f8ca1e0a0651d479ba7900907ecd01d4c07747`
  - `kv_top10_4slots_w3`: `a84698c77e3763a54a185a6f0b8787c76476297dc2287bc322c1941cad543e19`
  - `vector_layer12_name_w0.1`: `0c094fc90f1a82131f02bd59f7d98a01768db974a6095f9dd8f3765454cb355a`
  - `vector_layer12_name_w0.3`: `95a73eb1873b5584747115426dc769f63a68ecd28d500fabc3b6270cbe284758`
  - `vector_layer12_name_w1_seed43`: `d9abe31fcde142f56b818ba0fc39effd2639b03f5b7dace1b34c869b93019635`
  - `vector_layer12_name_w3`: `b39771c38e6245a819771026357549e93b4d8297171016916b0624c629a20e24`
- `analysis/analyse_sweep_and_two_hop.py`: all tables above.
