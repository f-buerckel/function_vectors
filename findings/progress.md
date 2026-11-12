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

## Next

Capacity of learned vectors (how many facts fit in one vector), and amortising them (a network mapping a document to its vector). See the context-difference README and [`ideas.md`](ideas.md).
