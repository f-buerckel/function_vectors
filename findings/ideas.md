# Ideas

Ideas to try later. Each entry names the commit it refers to.

## Corrupted prompts without contradicting answers (2026-11-12)

**Code at the time:** commit `e20deb4ece18581b6c01cb279e0a2385a55cb5fe` (branch `qwen-fv-extraction`)

**Problem with shuffled corruption.** The default corruption shuffles a person's answers. The query asks about one of the demonstrated facts, so the corrupted prompt usually states a wrong answer to the very question asked (for example "born in Oslo", then "where was she born?"). A head only scores if its patch overrides that explicit evidence, which is a different ability from supplying a missing fact, and not what the FV has to do later. In about a quarter of shuffles the queried answer stays in place, so the prompt isn't corrupted at all.

This differs from the paper: there, the query is a word that doesn't appear in the demonstrations, so shuffled labels hide the task without contradicting the query's answer.

**Options:**

1. **Other-person corruption.** Demonstrations from a person who shares none of the answers, so the queried answer is absent and nothing contradicts it. Implemented in `e20deb4` as `--corruption other_person` and running now with `--effect logprob --n_ie_trials 4`.
2. **Zero-shot corruption.** The bare question with no demonstrations. This is exactly the prompt the FV is injected into later, so it ranks heads by what we actually need. Not implemented yet.

**Catch for zero-shot.** The repo maps each token of the corrupted prompt to positions of a full ICL prompt with the same number of demonstrations (`compute_duplicated_labels` with `get_dummy_token_labels(n_shots)`). With no demonstrations, the last token would map to an early position, so the wrong mean activation would be patched without any error. The last token must be mapped explicitly to the dummy prompt's last slot (`query_predictive_token`).

**To compare:** run both corruptions on the same people, then compare the top-head overlap between them and their ranking stability.

## Per-person head selection (2026-11-12)

**Code at the time:** commit `e20deb4ece18581b6c01cb279e0a2385a55cb5fe` (branch `qwen-fv-extraction`)

**Current approach.** Following the paper, one set of top-10 heads is chosen from the indirect effects pooled over all people. Each person's FV sums that person's mean activations over these shared heads, so FVs differ through the activations, not the heads.

**Why question it.** The paper showed that a shared head set is robust for functions (tasks). That doesn't mean it holds for knowledge: the heads that carry one person's facts could differ from another's, and a shared set may miss them.

**Idea.** Rank heads separately for each person, using only that person's corrupted prompts, and build their FV from their own top 10.

**What it would show.** An upper bound on what shared heads lose. If per-person heads clearly beat shared ones, head choice depends on the knowledge and the shared-head assumption doesn't carry over from tasks.

**Caveats:**
- **Leakage:** selection uses the person's correct answers. Select with the extraction paraphrases and test only on held-out paraphrases.
- **Noise:** 2–4 prompts per person is too few to pick 10 of 1152 heads. In the 6 Nov run, per-person top-10 sets shared only 0–5 heads. This needs more corrupted prompts per person, at about a minute each.
- **Cost:** every new document would need its own indirect-effect sweep (1152 forward passes per prompt), instead of about 10 forward passes for mean activations. This matters for the PRAG comparison.
- **Comparability:** FVs built from different heads live in different coordinates, so cosine similarity, mean subtraction and the twin test become harder to interpret.

**To compare:** per-person against shared heads on the same people and held-out questions, once injection works.
