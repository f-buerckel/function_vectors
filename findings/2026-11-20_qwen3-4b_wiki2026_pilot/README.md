# Learned interventions on real documents: wiki2026 pilot (2026-11-20)

**Branch `document-interventions`.** First test of the learned interventions on real, longer documents the model can't have seen: 10 English Wikipedia articles about 2026 events (816–2,979 words), with questions written by subagents and filtered with the model (see `dataset_files/knowledge/wiki2026/README.md`). One intervention is learned per document, now with multi-token answers.

## Reproduce

- **Code:** commit `da7ac3749ccabfae12695a744051a342bfe5286b` on branch `document-interventions`, script `src/learn_document_intervention.py`.
- **Data:**
  - `dataset_files/knowledge/wiki2026/documents.jsonl`;
  - `filtered.json`: the 10 pilot documents, 147 questions after filtering;
  - per document 4–5 facts, each with 1–2 training phrasings and 1 test phrasing, plus 0–2 two-step questions;
  - in total 47 test phrasings and 10 two-step questions.
- **Commands:** `outputs/run.sh`. Each run is `learn_document_intervention.py --fast --early_stop` (up to 300 steps) with:

| Run | Options |
|---|---|
| `kv_top10_4slots_w{1,0}` | `--mode kv --lr 0.05 --n_slots 4` |
| `heads_last_w{1,0}` | `--mode heads --lr 0.5` (top-10 FV heads) |
| `layers_head_layers_w{1,0}` | `--mode layers --layers 24 26 29 31 32 33 34 35` (learning rate 0.1) |
| `vector_layer12_w{1,0}` | `--mode layers --layers 12` (learning rate 0.1) |

  `w1` and `w0` are `--contrast_weight 1` and `0`.
- **Analysis**, from the repo root: `python findings/2026-11-20_qwen3-4b_wiki2026_pilot/analysis/analyse_wiki2026_pilot.py findings/2026-11-20_qwen3-4b_wiki2026_pilot/outputs` (pandas only).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** 82 minutes for the 8 runs (6–17 minutes each, including evaluation). The GPU PC shut itself down afterwards.

## Setup

- **Training:**
  - **Input:** question + " " + answer. The loss is the cross-entropy of every answer token.
  - **Placement:** the intervention sits at the question's last token ("A:"), or for the key–value memory in the memory, which is visible from every position.
- **Contrast** (λ = 1), per step:
  - the 24 general contrast questions of experiment 10;
  - 24 questions sampled from the training questions of the next 6 documents.

  The KL to the unmodified model is taken at the first answer position.
- **Side effects:** each document's intervention is also applied to the test questions of the remaining 3 documents (held-out documents) and to 24 general questions (4 capitals plus the general questions of experiment 11).
- **Scoring:** greedy generation of up to 48 tokens. Right if the first line contains the answer or an alias (case, punctuation and accents ignored).
- **References from the filter:**
  - every kept question is answered *with* the document in context, so that's 100% by construction;
  - none is answered without it (here 1 of 47 test phrasings is, at the shorter generation length).

## Results

| Intervention | Numbers per document | λ | Training right | Test phrasings right (47) | Two-step right (10) | Held-out documents: own answer leaks | General questions right |
|---|---|---|---|---|---|---|---|
| Heads' 8 layers | 20,480 | 0 | 0.93 | **0.83** | 0.00 | 0.15 | 0.38 |
| | | 1 | 0.92 | **0.79** | 0.00 | 0.02 | 1.00 |
| Key–value memory | 10,250 | 0 | 0.93 | 0.70 | 0.00 | 0.06 | 0.50 |
| | | 1 | 0.98 | 0.51 | 0.00 | 0.00 | 1.00 |
| 10 head vectors | 1,280 | 0 | 0.94 | 0.68 | 0.00 | 0.11 | 0.31 |
| | | 1 | 0.96 | 0.49 | 0.00 | 0.01 | 0.99 |
| Single vector, layer 12 | 2,560 | 0 | 0.96 | 0.36 | 0.20 | 0.00 | 0.69 |
| | | 1 | 0.92 | 0.04 | 0.00 | 0.01 | 1.00 |
| No intervention | | | | 0.02 | 0.00 | | 1.00 |

- **With 47 test phrasings, a 95% interval is about ±13 points.**
- **Vectors in the heads' 8 layers recall trained facts in new wording best:** 79% with contrast, at no measurable cost to general questions and with 2% leakage into other documents' answers.
- **Without contrast,** every intervention damages general questions: only 31–69% stay right, against 100% without an intervention.
- **Contrast costs more here than on the toy data.** 8 layers 83 → 79%, but key–value memory 70 → 51%, head vectors 68 → 49%, and the single vector 36 → 4%.
- **Results vary strongly between documents.** For example, head vectors with contrast get 0% on d049 and 80% on d016 (per-document table in the analysis output).
- **Wrong answers are mostly near misses or mix-ups within the document:**
  - another fact's answer, e.g. "Noor Wali Mehsud" instead of "Hamdullah Fitrat";
  - wrong numbers, e.g. "3" for 31, "25,300" for 22,300;
  - a spelling variant: "Alhaji Salihu Umaru" for "Umar", counted wrong by the scorer.

**Two-step questions fail, but because of how they were written.**
- **No trained bridges:** in none of the 10 is the bridge entity (e.g. "Qatar" mediated the ceasefire, "Texas") one of the document's trained answers.
- **Different facts:** 7 of the 10 rest on facts no training question asks about at all.
- **What the model does:** it answers with a plausible but wrong country or city, e.g. "Pakistani rupee", "Phoenix".

So these questions measure whether an intervention knows document content **beyond its training questions**, and the answer is no. The question-writing instructions didn't require the bridge to be a trained fact.

## Interpretation

- **Learned interventions carry real-document facts.** Facts the model can't know reach up to 79–83% on new phrasings, with general questions intact under contrast training. That transfers the toy results to real documents, with multi-token answers.
- **They store what the training questions ask, not the document.** Facts not covered by a training question are not available (two-step bridges, 0/10). For RAG, where users ask anything about a document, this is the central limitation. The training data has to cover the document, not 5 facts.
- **Capacity matters more on real documents.** Under contrast, the 20,480-number layer vectors keep their accuracy, while the smaller interventions (heads, memory, single vector) lose 19–32 points.
  - A plausible reason, untested: on this data the contrast questions look exactly like the own questions ("In the 2026 X, which …?"), and only the subject differs. Telling them apart needs more capacity.
  - This reverses the toy result, where head vectors were the most efficient.
- **Limits:**
  - 10 documents and 47 test phrasings (±13 points); one seed;
  - only 1–2 training phrasings per fact;
  - two-step questions not usable as a test of reasoning;
  - side effects scored with 24 general questions.

## What we investigate next

1. **Cover the document:** more questions per document (all facts, not 5). Also a language-modelling loss on the document text itself, as PRAG trains on the document plus generated QA. Then test on facts no training question asked about.
2. **Fix two-step questions:** bridges must be trained facts. As written, they test something else.
3. **PRAG baseline:** a LoRA per document, trained on the same data, compared at equal side effects.
4. **Scale up:** questions for the other 54 documents once the training data design is settled.

## Files

- **`outputs/<run>/`:** `results.csv` (one row per question, with the modified and unmodified answers), `summary.json` and `run.log`. Plus `outputs/run.sh`, `outputs/timings.txt`, `outputs/runner.log` and `outputs/wiki2026_pilot.commit`.
- **`outputs/<run>/interventions.pt`** (git-ignored). SHA-256:
  - `heads_last_w0`: `a5302dbc49af4ff657a5effa50251daf94ab26f6ab5f33371ad38c8d092832a9`
  - `heads_last_w1`: `cf86fba46e1be1c298fcda88317481d666b00ba2bc17a22551fe9211095d6cba`
  - `kv_top10_4slots_w0`: `a656f89aba02d3e4985491d6c7bdec6706b852a2aa10457d1c06c1dabd41fd7a`
  - `kv_top10_4slots_w1`: `8e81c54ffa84243143a3a7899bf11df90049a2ad1407ec5c5f14bc5188207832`
  - `layers_head_layers_w0`: `632a0a2504489dcdd67a0e2245d2eccb0a9d5b71de166be1203db0021e1482be`
  - `layers_head_layers_w1`: `d584d8b470abe0761c2f9aa6a33f3d85f8b1b1bd5e68ad8b939176abff32cb89`
  - `vector_layer12_w0`: `e204f9edd622ea2795f617a4a8088cbb089be23ba058af4c3bbe141ccf833e98`
  - `vector_layer12_w1`: `2eb068bac2dcd71d113b96d94f094e9e5f919e49c8928910562c3642a500a206`
- **`analysis/analyse_wiki2026_pilot.py`:** the main table, per-document accuracy, the two-step bridge check and examples of wrong answers.
