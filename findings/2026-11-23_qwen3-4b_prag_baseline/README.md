# PRAG LoRA baseline on wiki2026 (2026-11-23)

**Branch `document-interventions`.** Follow-up to the [coverage experiment](../2026-11-20_qwen3-4b_wiki2026_coverage/README.md) (experiment 15). There, per-document vectors stored what their training questions asked and almost nothing else. This experiment trains **one LoRA per document**, following PRAG (Su et al., 2025), and asks whether a LoRA, or PRAG's model-written questions, reaches the held-out facts.

**Two of the four planned runs failed (out of GPU memory)**, including the one that would have trained our vectors on the model-written questions. So the effect of the questions is not separated from the effect of the LoRA.

## Reproduce

- **Code:** commit `192bf7a12f7ef0fb4df31b3754242fc589132155`, branch `document-interventions`. The four scripts on the GPU PC were identical to it (checked afterwards).
  - `src/augment_documents_prag.py`: passages, rewrites and question–answer pairs, written by Qwen3-4B.
  - `src/learn_document_lora.py`: the LoRA baseline. `src/document_data.py`: evaluation shared with the interventions.
- **Data:** `dataset_files/knowledge/wiki2026/filtered_v2.json` (70 trained facts, 30 held-out facts with 90 phrasings, 9 two-step questions) and `prag_augmented.json` (215 passages, 645 pairs, generated in this run).
- **Commands:** `outputs/run.sh`:

| Run | Options | Status |
|---|---|---|
| (data) | `augment_documents_prag.py` | done, 430 s |
| `vec_prag_layers8_all` | 8 layer vectors at every position, `--train_source prag`, contrast 1 | **failed** (OOM, first document) |
| `lora_prag` | `--data prag` | done, 2,954 s |
| `lora_qdoc_c1` | `--data questions --doc_lm_weight 1 --epochs 5 --contrast_weight 1` | done, 2,393 s |
| `lora_prag_c1` | `--data prag --contrast_weight 1` | **failed** (OOM, after 4 documents; nothing saved) |

- **Analysis**, from the repo root: `python findings/2026-11-23_qwen3-4b_prag_baseline/analysis/analyse_prag_baseline.py findings/2026-11-23_qwen3-4b_prag_baseline/outputs` (pandas; also reads the experiment-15 outputs).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, peft 0.17.1, float16 base weights with float32 LoRA parameters.
- **Interruptions:**
  - `vec_prag_layers8_all`: the own-question batch holds all of a document's generated pairs (87 for d002, against about 14 subagent questions), at once and with gradients.
  - `lora_prag_c1`: the reference logits for the contrast documents' questions are computed in one batch (477 prompts for the fifth document, d031).
  - The queue carried on past both failures. The PC shut itself down afterwards, as requested.

## Setup

- **LoRA**, from PRAG's official code (read through a subagent's report; the code itself has no licence and was not copied): rank 2, alpha 32, no dropout, on `gate_proj`, `up_proj`, `down_proj` of every layer; AdamW, learning rate 3e-4, batch size 1. 2,654,208 numbers per document, 130 times the 20,480 of the 8 layer vectors.
- **`lora_prag` data:** 5 chat-format examples per passage, loss on the whole sequence, 1 epoch (60–165 steps per document). No contrast.
- **`lora_qdoc_c1` data:** as `qa_doc_layers8_all` in experiment 15 (subagent questions + document text), 5 epochs (85–160 steps), contrast weight 1.
- **Deviations from PRAG:**
  - one LoRA per document instead of per 100-word passage; passages of up to 120 words from one article;
  - our own prompt wording (PRAG's was known only partly), with questions required to name the subject; the closed-book chat prompt is a guess at PRAG's;
  - float16 base weights.
- **Evaluation:** as in experiment 15, in the Q:/A: format ("qa") and in PRAG's chat format without passages ("chat").
- **Not tuned or matched:** learning rate, epochs and the contrast sampling (8 + 8 questions per step, against 24 + 24 for the vectors).

## Results

Intervals are 95% bootstrap intervals over facts. The experiment-15 rows are for comparison.

| Run | Own training items right | Test phrasings (70) | Held-out phrasings (90) | Held-out facts with at least one right (30) | Two-step (9) | Leak into other documents | General right (240) | General unchanged |
|---|---|---|---|---|---|---|---|---|
| Exp. 15 `qa_layers8_all` (vectors, questions) | 0.91 | 46 (0.66) | 0 | 0 | 3 | 0.10 | 0.99 | 0.47 |
| Exp. 15 `qa_doc_layers8_all` (vectors, questions + text) | 0.91 | 31 (0.44) | 7 (0.08; 0.02–0.16) | 4 | 0 | 0.05 | 0.86 | 0.22 |
| `lora_qdoc_c1`, qa | 0.86 | **56 (0.80)** | 7 (0.08; 0.01–0.17) | 4 | 2 | 0.11 | 0.99 | 0.54 |
| `lora_qdoc_c1`, chat | 0.69 | 43 (0.61) | 9 (0.10; 0.02–0.19) | 5 | 2 | 0.05 | 1.00 | 0.23 |
| `lora_prag`, qa | 0.18 | 17 (0.24) | **24 (0.27; 0.13–0.42)** | 10 | 0 | 0.06 | 1.00 | 0.09 |
| `lora_prag`, chat | 0.20 | 17 (0.24) | **24 (0.27; 0.13–0.41)** | 11 | 0 | 0.04 | 1.00 | 0.00 |

"Own training items" differ by run: the generated pairs for `lora_prag`, the subagent training questions otherwise.

### 1. LoRA and vectors on the same data

- **The LoRA stores the trained facts better:** 80% against 44% of test phrasings (best vectors overall, without text: 66%).
- **It keeps general questions right:** 99% against 86%, and doesn't give garbled answers.
- **It reaches held-out facts no more often:** 7 of 90 for both.
- **Two-step:** 2/9 against 0/9. Too few questions to compare.
- **Leakage into other documents is higher:** 11% against 5%.

### 2. PRAG's data

- **Held-out facts: 24 of 90**, the most of any run so far. These are phrasings of 10–11 of 30 facts, so the interval is wide (0.13–0.42).
- **"Held-out" means no subagent question, not unseen:** the generated pairs were made from the same text. By a loose answer-matching heuristic they cover 20 of the 30 held-out and 43 of the 70 trained facts.
  - Covered held-out facts: 21 of 60 phrasings right. Uncovered: 3 of 30 (qa format). In chat format 19 of 60 and 5 of 30.
- **Training is too short to be a fair ceiling.** The LoRA answers only 18–20% of its own generated pairs. Its test accuracy (24%) is about the same as its held-out accuracy (27%), as expected if neither set was trained on directly.
- **Side effects are hidden by the scoring:** 100% of general answers contain the right answer, but only 9% are unchanged (qa). In the qa format, 67 of 240 general answers go on to mention words from the document's title ("Paris. Question: In the 2026 Afghanistan–Pakistan war, how many people were injured in…"). Without contrast there was nothing to prevent this. In chat format it is 6 of 240. `lora_qdoc_c1` (with contrast): 0.

### 3. Scoring check

An answer is right if its first line contains the reference anywhere, and long answers get more chances. Counting only the first 12 words (qa format): `lora_prag` 24 → 22 held-out, `lora_qdoc_c1` 7 → 6 and 56 → 56 test, experiment 15's `qa_doc_layers8_all` 7 → 2 held-out and 31 → 28 test. (Not meaningful in chat format, where answers often restate the question first.)

## Interpretation

- **A LoRA is the stronger store for trained facts,** at 130 times the size. With the data held fixed it has higher accuracy and fewer general errors than the 8 layer vectors, but more leakage.
- **On the same data, the LoRA doesn't fix the held-out facts (8% for both).** So the limit in experiment 15 isn't specific to vectors. Raw document text in the loss doesn't make facts answerable for a LoRA of this size either, with this training budget.
- **Model-written questions look promising, but this isn't shown yet.** `lora_prag` differs from `lora_qdoc_c1` in data, format, loss, number of steps and contrast all at once. The run that would isolate the questions (vectors on the same pairs) failed.
- **Limits:**
  - 10 documents, one seed; held-out results rest on 10–11 facts;
  - training budgets and learning rates not matched or tuned, so neither "LoRA beats vectors" nor `lora_prag`'s numbers are final;
  - loss scaling can skip optimizer steps with non-finite gradients; how many was not logged;
  - PRAG reimplemented from a report of its code and adapted to one LoRA per document, so this is not PRAG's published result;
  - the coverage heuristic is loose.

## What we investigate next

1. **Fix the memory use** (own questions in minibatches; contrast reference logits in batches) and rerun `vec_prag_layers8_all` and `lora_prag_c1`. Needs the GPU; not started.
2. **Isolate the questions:** train the LoRA and the vectors on the generated pairs alone, in the same format, with enough epochs to fit them.
3. **Match training budgets,** and choose the LoRA's epochs by training-item accuracy, so that the comparison is at equal fit.

## Files

- **`outputs/lora_prag/` and `outputs/lora_qdoc_c1/`:** `results.csv` (one row per question, format and document), `summary.json`, `run.log`.
- **`outputs/vec_prag_layers8_all/run.log`, `outputs/lora_prag_c1/run.log`:** the failed runs.
- **`outputs/run.sh`, `timings.txt`, `augment.log`, `shutdown_watch.*`, `prag_baseline.commit`.**
- **`outputs/<run>/loras.pt`** (git-ignored). SHA-256:
  - `lora_prag`: `5d6348e52648b6a25c6237d42d03e0ca52722dab9b1379849f85cd37333327c6`
  - `lora_qdoc_c1`: `81298545e4b3ef3cf20ff50e7ac72553920f43678f4224d6e72951e57947aa04`
- **`dataset_files/knowledge/wiki2026/prag_augmented.json`:** the generated passages, rewrites and pairs.
- **`analysis/analyse_prag_baseline.py`:** all tables above.
