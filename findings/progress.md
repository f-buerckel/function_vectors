# Progress

One short entry per experiment, newest last. Details are in each experiment's `README.md`.

**Question:** can a function vector (FV) carry document knowledge, such as facts about a fictional person, so the model answers without the document in context?

## 1. Extraction setup run (2026-11-06)

[`2026-11-06_qwen3-4b_extraction_smoke`](2026-11-06_qwen3-4b_extraction_smoke/README.md) · commit `31d6170`

- **What:** first run of the paper's FV extraction on Qwen3-4B, with 6 fictional people × 4 city facts. Corruption shuffles the answers, and the effect is a change in probability (paper defaults).
- **Results:**
  - The model uses the facts in context (23/24 right) and doesn't know them without it (0/24).
  - The head ranking is unusable: 1 of 12 prompts gives 99% of the top-10 effect.
  - FVs of people with the same cities in swapped roles are near-identical (cosine 0.89–0.97, others 0.36–0.49).

## 2. Stable head ranking (2026-11-12)

[`2026-11-12_qwen3-4b_extraction_logprob_otherperson`](2026-11-12_qwen3-4b_extraction_logprob_otherperson/README.md) · commit `e20deb4`

- **Changed from 1:**
  - The effect is a change in log-probability.
  - Corruption uses another person's facts instead of shuffling.
  - There are 4 prompts per person instead of 2.
- **Results:**
  - The ranking is stable: largest single-prompt share 0.07 (was 0.99), split-half top-10 overlap 7/10.
  - All top heads are in layers 24–35.
  - Same-city FVs are still near-identical (0.86–0.97).

## 3. First injection (2026-11-12)

[`2026-11-12_qwen3-4b_injection_smoke`](2026-11-12_qwen3-4b_injection_smoke/README.md) · commit `95c9eb6`

- **Changed from 2:** run 2's FVs are added to bare questions, over 13 layers × 5 scales. They are compared with a same-cities partner's FV, a different-cities person's FV and a random vector.
- **Results:**
  - From scale 2 upward, the own FV gets about 25% right at nearly every layer (best 29%; no vector 0%, random 4%).
  - It pushes one of the person's cities, the same one for every question. The partner's FV does as well (25%), so there's content but no binding.
  - The pushed city appears to follow which fact the extraction prompts asked about most (4 of 6 people).

## 4. Balanced and per-fact FVs (2026-11-12)

[`2026-11-12_qwen3-4b_balanced_and_per_fact_fvs`](2026-11-12_qwen3-4b_balanced_and_per_fact_fvs/README.md) · commit `3b4d08c`

- **Changed from 3:** same heads, different averaging of mean activations.
  - **Balanced FV:** every fact counts equally.
  - **Per-fact FV:** averaged over one fact's prompts only, and injected for that fact's question.
- **Results:**
  - Balanced FVs are still about 25% right and push one city per person. Balancing changed which city wins, not that one wins.
  - The per-fact FV for the fact asked is 100% right from scale 2 upward at nearly every layer. A per-fact FV for a different fact always gives its own city (0% right), so the question never wins.
  - Per-fact FVs encode the city: 0.95 cosine for the same city across different people, 0.56 for one person's different cities.
  - So the averaged FV captures the answer the model was about to give, not knowledge the question can query.

## 5. Learned vector (2026-11-12)

[`2026-11-12_qwen3-4b_learned_vector`](2026-11-12_qwen3-4b_learned_vector/README.md) · commit `56a3dad`

- **Changed from 4:** no extraction. One vector per person is optimised by gradient descent so the frozen model answers that person's 12 training questions. It is added at the last token or at the person's name, at layers 6, 12, 18 and 24.
- **Results:**
  - Training questions are 100% right. Held-out paraphrases are 67–88% right, against 25% for a bag of answers, so one vector can hold 4 bound facts.
  - All held-out errors are another of the person's cities. The "birthplace" paraphrase is the hardest.
  - Last-token vectors answer by question type whoever is asked about, and harm unrelated questions. Name vectors at layers 6 and 12 leave unrelated questions intact (96–100%).
  - So capacity isn't the problem; FV extraction is.

## 6. Context-difference vector (2026-11-12)

[`2026-11-12_qwen3-4b_context_diff_vector`](2026-11-12_qwen3-4b_context_diff_vector/README.md) · commit `56af012`

- **Changed from 5:** no training. The vector is the mean difference in the hidden state with and without the facts in context, taken at the person's name (or the last token), and added there in bare questions.
- **Results:**
  - At the name, 0% right at every layer and scale. At the last token, only layer 24 does anything, and that's the bag-of-answers pattern again (up to 29%).
  - The last-token difference is 95–99% shared by all people. The name difference is about half person-specific, but that part is nearly orthogonal to the working learned vectors (cosine 0.01–0.11).
  - So far, only optimisation gives a working vector; simple extraction doesn't.

## 7. Capacity of learned vectors (2026-11-12)

[`2026-11-12_qwen3-4b_learned_vector_capacity`](2026-11-12_qwen3-4b_learned_vector_capacity/README.md) · commit `b3eed6b`

- **Changed from 5:** a new setup dataset with 8 people × 8 city facts. Learned vectors hold 2, 4 or 8 facts, at the last token or the name, at layers 6 and 12.
- **Results:**
  - Training questions are 100% right for every number of facts.
  - Held-out accuracy falls from 81–94% (2 facts) to 72–84% (4) and 58–77% (8), still 4.6–6.2 times the bag-of-answers baseline at 8 facts.
  - The same facts get worse as more are added (born city 81% → 56% → 44%), and 115 of 116 errors are confusions with the vector's other facts.
  - So one vector fits at least 8 facts, but generalisation to new phrasings suffers from interference.

## 8. Learned vectors per head and per layer (2026-11-12)

[`2026-11-12_qwen3-4b_learned_heads_and_layers`](2026-11-12_qwen3-4b_learned_heads_and_layers/README.md) · commit `e69cbd7`

- **Changed from 7:** instead of one vector, learn one per top-10 FV head (128 numbers each, added to the head's output), or one per layer for several layers. 8 facts per person.
- **Results:**
  - At the last token, every variant beats the single vector on held-out questions (single 69–77%): 10 heads reach 83–92% with 1,280 numbers, and the heads' 8 layers reach 100%.
  - But the most accurate ones answer unrelated questions with the person's cities: capitals drop to 0% ("capital of France?" → "Lisbon"). The single vector keeps 66–81%.
  - So head selection finds an efficient place to intervene, but there the vector overrides every question. Held-out accuracy and side effects have to be measured together.

## 9. Learned key–value memory for heads (2026-11-12, branch `kv-memory`)

[`2026-11-12_qwen3-4b_kv_memory`](2026-11-12_qwen3-4b_kv_memory/README.md) · commit `d95b569` (analysis `5632189`)

- **Changed from 8:** instead of fixed head vectors, each selected head gets learned key–value entries that its own query reads from, like extra tokens in its context.
- **Results:**
  - At the top-10 FV heads it's as accurate as head vectors (88–91% held-out with 8 facts), and capitals again drop to 0%.
  - The memory takes about 80% of those heads' attention whatever is asked, so a person's memory answers other people's questions with that person's cities (100%).
  - In early layers it preserves capitals (75%) but generalises worse (52%).
  - Question-dependent retrieval doesn't emerge, because every training question is about the memory's owner.

## 10. Contrast training (2026-11-15, branch `contrast-training`)

[`2026-11-15_qwen3-4b_contrast_training`](2026-11-15_qwen3-4b_contrast_training/README.md) · commit `5344331`

- **Changed from 7–9:** training adds contrast questions (4 other people's questions and general questions), where a KL loss keeps the model's predictions unchanged. Five interventions, each trained with λ = 0 and λ = 1.
- **Results:**
  - Contrast removes the override. Held-out questions about 3 people never used as contrast get the owner's cities at most 1% of the time (was 45–100%), capitals are never answered with a dataset city, and KL to the unmodified model drops 40–600 times.
  - Held-out accuracy falls 6–16 points, so part of the earlier accuracy was the override. Head vectors 92 → 86%, the heads' 8 layers 100 → 84%, KV memory 88 → 75%, single vectors 72 → 58% (last token) and 66 → 59% (name).
  - At equal side effects, the 10 head vectors are the most accurate per number (86% with 1,280 numbers; bag-of-answers baseline 12.5%). The name vector has the smallest side effects but 59%.

## 11. Contrast-weight sweep and two-hop questions (2026-11-15/16, branch `contrast-training`)

[`2026-11-15_qwen3-4b_contrast_sweep_and_two_hop`](2026-11-15_qwen3-4b_contrast_sweep_and_two_hop/README.md) · commits `5344331` (sweep), `5a25380` (evaluation)

- **Changed from 10:**
  - λ ∈ {0.1, 0.3, 3} for head vectors, the name vector and the key–value memory, plus a second seed.
  - All 21 runs are scored on full generated answers, with two-hop questions ("In which country was X born?"), more general questions, and the memory's attention.
- **Results:**
  - λ = 0.1 already removes most of the override. Larger λ trades held-out accuracy for fewer side effects; a second seed moves accuracy by 3–5 points.
  - Two-hop at λ = 1: the heads' 8 layers 81%, key–value memory 70%, head vectors 44%, single vectors 31–33% (in context 100%, unmodified 2%, bag 12.5%). Head vectors lose two-hop ability as λ grows (91% → 28%); the memory gains it (23% → 70%).
  - With full answers, capitals are 100% right in every contrast run; first-token scoring had understated this.
  - With contrast, the memory is read depending on the question (λ = 1: 67% attention for the owner's questions, 55% for other people's, 28% for general ones; λ = 0: 79–80% for all).

## 12. Head selection by binding effect (2026-11-16, branch `binding-effect`)

[`2026-11-16_qwen3-4b_binding_effect`](2026-11-16_qwen3-4b_binding_effect/README.md) · commit `416d60a`

- **Changed from 2–3:** heads are ranked by how much they raise the right city *relative to the person's other cities* (binding effect), instead of the right city's log-probability. Then the same FV extraction and injection test.
- **Results:**
  - The ranking changes partly: 4 of the top 10 heads are shared, 2 new ones are in layers 20–23, and the rest are still in layers 28–33. The ranking is unstable (split-half overlap 2/10, was 7/10).
  - Injection doesn't improve: best cell 29%, and own FV 21% against 22% for the same-cities partner. Each person's FV still pushes one city for every question.
  - Not a decisive test, because 4 corrupted prompts per person don't give a stable binding ranking.

## 13. Binding effect with the paper's prompt counts, mid-layer heads, faster training (2026-11-16, branch `fast-training`)

[`2026-11-16_qwen3-4b_binding_paper_counts`](2026-11-16_qwen3-4b_binding_paper_counts/README.md) · commits `6fb20c9`, `a01618f`, `ba7be56`

- **Changed from 12:** 100 mean and 25 corrupted prompts per person (the paper's counts), for both the binding and the log-prob ranking; FVs from heads below layer 24 only; and the training speed-ups `--fast` / `--early_stop`.
- **Results:**
  - The binding ranking gets more stable (split-half 4/10, was 2/10), the log-prob one very stable (9/10). Both FVs still push one city per person: own 19–22%, twin 20–23%.
  - FVs from heads below layer 24 do as little as a random vector (3%).
  - More data doesn't move the 25% ceiling. The extraction line is closed.
  - The batched indirect effect is 6–7× faster with the same rankings. `--fast` reproduces experiment 10 exactly; with early stopping, training is 2.4–3.4× faster within seed noise.

## 14. Learned interventions on real documents: wiki2026 pilot (2026-11-20, branch `document-interventions`)

[`2026-11-20_qwen3-4b_wiki2026_pilot`](2026-11-20_qwen3-4b_wiki2026_pilot/README.md) · commit `da7ac37`

- **Changed from 10–11:**
  - 10 Wikipedia articles about 2026 events (816–2,979 words) instead of fictional people; 147 subagent-written, model-filtered questions.
  - Multi-token answers: the loss over all answer tokens, scoring on generated answers.
- **Results:**
  - Test phrasings, with contrast: heads' 8 layers 79%, key–value memory 51%, head vectors 49%, single vector 4% (no intervention 2%). General questions stay right (99–100%), and leakage into other documents is 0–2%.
  - Contrast costs the smaller interventions 19–32 points. On real documents, capacity matters more than on the toy data.
  - Two-step questions: 0/10. Their bridge facts were never trained, so the interventions store only what the training questions ask, not the document.

## 15. Covering the document: held-out facts, document-text loss, two-step over trained facts (2026-11-20, branch `document-interventions`)

[`2026-11-20_qwen3-4b_wiki2026_coverage`](2026-11-20_qwen3-4b_wiki2026_coverage/README.md) · commits `e5d0329`, `058338e`, `104fce7`

- **Changed from 14:**
  - The same 10 documents with 12 facts each, covering the whole text. 30 facts (90 phrasings) are held out of training, and every two-step question's bridge is a trained fact.
  - A language-modelling loss on the document text (as in PRAG), and vectors added at every position.
  - 6 runs, all with contrast; the memory with questions plus text failed three times (out of memory, PC crash, hang) and is left out.
- **Results:**
  - Trained on questions only, no intervention answers any held-out fact (0/90). Test phrasings: 66% for the 8 layer vectors at every position, 51% at the last token, 50% for the memory.
  - The text loss gives 8% of held-out facts (7/90) with the 8 layer vectors, with or without questions. But general questions drop to 86–87% (garbled answers) and trained facts to 44%. The memory learnt nothing from text (loss flat from the first step).
  - Memorising is not answering: on 5 articles the text-only vectors learn the text almost word for word (loss ≤ 0.10), yet answer 2 of 42 held-out and 1 of 33 trained-fact phrasings.
  - Two-step questions: 3/9 at best (unmodified 0/9), although the bridge fact is answered right in 9/9.

## Next

A PRAG LoRA baseline on the same data (document text and questions); dense, possibly model-generated questions covering every sentence; then the remaining 54 wiki2026 documents with more two-step questions. See the coverage README.
