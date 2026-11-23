# Covering the document: held-out facts, document-text loss and two-step questions over trained facts (2026-11-20)

**Branch `document-interventions`.** Follow-up to the [wiki2026 pilot](../2026-11-20_qwen3-4b_wiki2026_pilot/README.md) (experiment 14). The pilot found that a per-document intervention stores only the facts its training questions ask about. This experiment asks:

1. Does training on the **document text itself** (a language-modelling loss, as in PRAG) make facts available that no training question asked about?
2. Do **two-step questions** work when their bridge is a trained fact?

## Reproduce

- **Code** (branch `document-interventions`, script `src/learn_document_intervention.py`):

| Commit | Runs |
|---|---|
| `e5d0329c7fd0c95ef8ac60247e6b46b546f07bcf` | `qa_layers8_last`, `qa_layers8_all` |
| `058338e6bad57e1893be223f64544996d3286b68` | `qa_kv`, `qa_doc_layers8_all`, `doc_layers8_all`, first attempt at `qa_doc_kv` |
| `104fce74833da695688bfce95875d9616b485845` | `doc_kv`, second and third attempts at `qa_doc_kv` |

  - `058338e` only saves memory: reference logits only at the last token, and freeing GPU memory between documents.
  - `104fce7` backpropagates each loss part separately. That gives the same gradient with less memory in use.
- **Data:**
  - `dataset_files/knowledge/wiki2026/questions_v2/`: the same 10 pilot documents, rewritten with `question_instructions_v2.md`. 12 facts per document cover the whole text, plus 3 two-step questions, each naming its bridge fact.
  - Filtered with `src/filter_wiki2026_questions.py --version 2` into `filtered_v2.json`: 100 facts and 9 two-step questions kept.
  - **Trained facts:** 70, with 139 training phrasings and 70 test phrasings. Every bridge fact is trained.
  - **Held-out facts:** 30. Their 90 phrasings (2 former training phrasings plus the test phrasing) are never used for training.
- **Commands:** `outputs/run.sh` (first two runs), `outputs/run_part2.sh`, `outputs/run_part3.sh`. Every run uses `--questions filtered_v2.json --fast --early_stop --contrast_weight 1` (up to 300 steps).

| Run | Options | Trained on |
|---|---|---|
| `qa_layers8_last` | `--mode layers --layers 24 26 29 31 32 33 34 35 --position last` | questions |
| `qa_layers8_all` | same layers, `--position all` | questions |
| `qa_kv` | `--mode kv --lr 0.05` | questions |
| `qa_doc_layers8_all` | 8 layers, `--position all --doc_lm_weight 1` | questions + document text |
| `doc_layers8_all` | 8 layers, `--position all --qa_weight 0 --doc_lm_weight 1` | document text |
| `doc_kv` | `--mode kv --lr 0.05 --qa_weight 0 --doc_lm_weight 1` | document text |
| `qa_doc_kv` | `--mode kv --lr 0.05 --doc_lm_weight 1` | **not finished**, see below |

- **Analysis**, from the repo root: `python findings/2026-11-20_qwen3-4b_wiki2026_coverage/analysis/analyse_wiki2026_coverage.py findings/2026-11-20_qwen3-4b_wiki2026_coverage/outputs` (pandas only).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** 116 minutes for the 6 finished runs, 14–27 minutes each including evaluation (`outputs/timings.txt`).

### Interruptions: `qa_doc_kv` failed three times

1. **Out of GPU memory** on the first document at `058338e`, because all loss graphs were held at once. Fixed by `104fce7`. Log: `outputs/qa_doc_kv_oom_attempt/`.
2. **The whole PC crashed** during the second document at `104fce7`. Log: `outputs/qa_doc_kv_crash_attempt/`.
3. **Restart hung:** 4 documents finished, then the process ran at 300% CPU for 3.5 hours without finishing the fifth document. I stopped it; afterwards the GPU worked normally. Log: `outputs/qa_doc_kv_hang_attempt/`. Its timings line (14,007 s) is this attempt.

None of the other runs failed, so this combination (memory plus document text, the most memory-hungry) is the likely trigger. The run is left out.

## Setup

What changed from experiment 14:
- **Data:** 12 facts per document instead of 4–5, covering every section. About 30% of facts are held out completely.
- **Document-text loss** (`--doc_lm_weight`): each step samples 4 chunks of 256 tokens from the document's title and text. The loss is next-token cross-entropy over each chunk, with the intervention applied at every position (`--position all`) or, for the memory, visible from every position.
- **Every-position vectors** (`--position all`): the 8 layer vectors are added at every token instead of only the last one. That's needed for the document-text loss, and is also tested without it.
- **Contrast** stays as in experiment 14, at weight 1 for every run. The KL to the unmodified model is taken on general and other documents' questions.
- **Scoring:** as in experiment 14, greedy generation up to 48 tokens. Right if the first line contains the answer or an alias.

## Results

| Run | Trained on | Training right | Test phrasings right (70) | Held-out facts right (90) | Two-step right (9) | Held-out documents: own answer leaks | General questions right (240) | Mean steps |
|---|---|---|---|---|---|---|---|---|
| `qa_layers8_last` | questions | 0.90 | 0.51 | 0.00 | 1 | 0.04 | 1.00 | 159 |
| `qa_layers8_all` | questions | 0.91 | **0.66** | 0.00 | **3** | 0.10 | 0.99 | 126 |
| `qa_kv` | questions | 0.99 | 0.50 | 0.00 | 0 | 0.03 | 1.00 | 221 |
| `qa_doc_layers8_all` | questions + text | 0.91 | 0.44 | **0.08** | 0 | 0.05 | 0.86 | 278 |
| `doc_layers8_all` | text | 0.07 | 0.07 | **0.08** | 0 | 0.01 | 0.87 | 282 |
| `doc_kv` | text | 0.01 | 0.03 | 0.00 | 0 | 0.01 | 1.00 | 71 |
| No intervention | | | 0.03 | 0.00 | 0 | | 1.00 | |

- **With 70 test phrasings, a 95% interval is about ±12 points; with 90 held-out phrasings, 8% is 7 answers.**

### 1. Held-out facts

- **Questions alone store nothing beyond the questions:** 0 of 90 for all three runs, now with facts drawn from the same document sections as the trained ones.
- **The document-text loss gives a small effect:** 7 of 90 phrasings with and without training questions.
  - With questions: 4 facts, e.g. "Operation Savanna Shield", "Anutin Charnvirakul", "Top 14", "$50 million".
  - Text alone: 7 phrasings of 6 facts, e.g. "Operation Absolute Resolve", "La Catalina". One is a lucky list ("…25 million dollars, 15 million dollars, 10 million dollars…"), so 6 phrasings of 5 facts are real.
  - **No overlap with training data:** none of these answers appears in any trained fact's question or evidence.
- **The key–value memory learns nothing from the text:** 0/90, and it changes almost no answers (general answers unchanged 0.98).
  - Its text loss is flat from the first step: the mean of the first 10 steps is 2.39–3.77, of the last 10 steps 2.33–3.69 (`outputs/loss_curves.csv`).
  - Early stopping ended it after 60–120 steps, correctly: the loss wasn't falling, so more steps would not have helped.
- **Text alone barely stores the trained facts either:** 7% of test phrasings.

### 2. Memorising the text is not the same as answering questions about it

The 8 layer vectors trained on document text only (`doc_layers8_all`) learn some articles almost word for word:

| Document | Text loss, first 10 steps | Text loss, last 10 steps | Held-out facts right | Test phrasings right |
|---|---|---|---|---|
| d031 | 2.24 | **0.03** | 0.00 | 0.00 |
| d032 | 2.01 | **0.05** | 0.11 | 0.00 |
| d025 | 2.31 | **0.08** | 0.17 | 0.00 |
| d016 | 2.75 | **0.09** | 0.00 | 0.12 |
| d026 | 2.59 | **0.10** | 0.00 | 0.00 |
| d048 | 2.61 | 0.46 | 0.00 | 0.00 |
| d056 | 2.54 | 0.71 | 0.11 | 0.14 |
| d050 | 2.37 | 0.85 | 0.44 | 0.12 |
| d002 | 2.45 | 1.03 | 0.00 | 0.14 |
| d049 | 3.41 | 1.26 | 0.00 | 0.12 |

- **Learned almost word for word:** on the 5 articles with a final loss of 0.10 or less, the vectors predict the next token of a 256-token piece of the article with about 90% probability on average (e^−0.1).
- **Yet they rarely answer questions about it:** on those 5 articles, 2 of 42 held-out-fact phrasings and 1 of 33 test phrasings are right.
- **Memorising doesn't predict answering:** the less well-memorised articles do no worse (d050: 44% of held-out facts at loss 0.85).
- **Interpretation:** an intervention that can reproduce an article as text can't use the same knowledge to answer a question about it. Reaching a fact from a question is a different route from continuing the sentence that contains it.
  - This matches findings on knowledge in language models: models only learn to extract a fact from a question when it is seen in many phrasings (Allen-Zhu & Li, "Physics of Language Models, Part 3.1"), and learn facts only in the direction they're written (the "reversal curse", Berglund et al.).
  - **Consequence for this project:** question-style training data, not the raw text, is what makes knowledge usable. This is also why PRAG adds generated question–answer pairs to the document text.

### 3. Side effects of the document-text loss

- **General questions:** 86–87% right with document text in the 8 layer vectors (33 and 32 of 240 wrong), against 99–100% without. Only 17–22% of general answers stay unchanged.
  - The errors are garbled: "Albert N. W. S. E. H. H. H…", "Mount Kilgour", "A: A: A: A:…", "The largest mammon is the blueyrog".
  - So contrast training on short questions does not protect the model from a vector added at every position that has learnt to continue a document.
- **Trained facts suffer when text is added:** test phrasings 66% → 44% (`qa_layers8_all` → `qa_doc_layers8_all`).

### 4. Two-step questions over trained facts

| Run | Two-step right | Bridge fact's test phrasing right |
|---|---|---|
| `qa_layers8_last` | 1/9 | 8/9 |
| `qa_layers8_all` | 3/9 | **9/9** |
| `qa_kv` | 0/9 | 5/9 |
| `qa_doc_layers8_all` | 0/9 | 8/9 |

- **Right with `qa_layers8_all`:** Ankara (bridge: Turkey), Swiss franc (bridge: the Swiss system) and Northamptonshire (bridge: Northampton Saints). Unmodified: 0/9.
  - One of these three may be guessable: "a format named after a country" suggests Switzerland, though the unmodified model gets it wrong.
- **Bridges are known but rarely used:** in 9/9 cases `qa_layers8_all` answers the bridge question right, yet only 3 two-step questions.
  - Typical failure: the answer to the first step instead of the second, e.g. "Amnesty International" (the bridge) instead of its headquarters city, London.
  - Or a plausible wrong country: "Caracas" instead of Bogotá, "Mexican peso" instead of yen.
- **Every-position vectors help two-step:** 3/9 against 1/9 at the last token. That fits the toy result (experiment 11: two-hop 81% for the heads' 8 layers), where the fact also had to be used at a token other than the answer. With 9 questions, this is a hint, not a result.

### Every-position vs last-token vectors (questions only)

- **Test phrasings 66% vs 51%,** two-step 3/9 vs 1/9.
- **Leakage into other documents rises:** 10% vs 4% of held-out documents' questions get this document's answer.

## Interpretation

- **The central limitation of experiment 14 holds with full-coverage questions:** an intervention trained on questions answers those facts in new wording (up to 66%), but nothing else in the document (0/90).
- **Memorising an article doesn't make its facts answerable.** Vectors that reproduce an article almost word for word answer almost none of its questions (section 2).
- **The document-text loss, PRAG's main ingredient, helps only a little here.** 8% of held-out facts, at a high cost: general questions 86% and trained facts 44%. With 20,480 numbers per document, predicting the text word for word competes with answering questions.
- **Two-step reasoning over a stored fact is possible but weak:** 3/9 when the bridge is known in 9/9. On the toy data it was 81%. Real bridges (organisations, events) are less direct than "person → city".
- **Limits:**
  - 10 documents and one seed;
  - 9 two-step questions;
  - the memory with questions plus document text never finished;
  - only one weight for the text loss (1) and one chunk size (256).

## What we investigate next

1. **PRAG LoRA baseline** on the same data (document text plus questions): does a LoRA store held-out facts better, and at what side effects? If it also fails on held-out facts, the limit is training on text, not vectors.
2. **Dense question coverage:** many generated questions per document, covering every sentence, possibly written by the model itself, to test whether question training reaches the held-out facts once they're asked about.
3. **More two-step questions** when scaling to the other 54 documents.

Not pursued: a sweep over smaller text-loss weights. It could only protect a gain of about 8%.

## Files

- **`outputs/<run>/`:** `results.csv` (one row per question, with modified and unmodified answers), `summary.json` and `run.log`.
  - `outputs/qa_doc_kv_{oom,crash,hang}_attempt/run.log`: the failed runs.
  - Plus `outputs/run*.sh`, `outputs/runner*.log`, `outputs/timings.txt` and `outputs/wiki2026_coverage*.commit`.
- **`outputs/<run>/interventions.pt`** (git-ignored). SHA-256:
  - `qa_layers8_last`: `98689c84525b7a305ca1b981655971f9fc79d98c075cec4ae8b4d966b72991d3`
  - `qa_layers8_all`: `f95e4007608dd9fa390e583b5b845f4635f8b006a5f4f7df83f5e7df8c560128`
  - `qa_kv`: `af63c791f079b85614732ecfbcae69b365e3b19dc403a040595e0c0f9400c057`
  - `qa_doc_layers8_all`: `8e314bc1820c395e82a209c52f287a21a58a972a1168458695b2fe810875e0d3`
  - `doc_layers8_all`: `7bfefbe37d651fe96377c85d8a3c4d3d8f1e1e403e6700ab65941fa66f89c38f`
  - `doc_kv`: `d141b4d0ec2b9b3dca624f27db8376d992c21645c47514214d83168b74cd2d00`
- **`outputs/loss_curves.csv`:** own and contrast loss per step, run and document, exported from the `interventions.pt` files.
- **`analysis/analyse_wiki2026_coverage.py`:** the main table, text loss against accuracy, steps and loss per document, held-out facts answered right, two-step questions with their bridges, per-document accuracy, and wrong general answers.
