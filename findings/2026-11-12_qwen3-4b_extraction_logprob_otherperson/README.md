# FV extraction with log-probability effect and other-person corruption (2026-11-12)

A rerun of the [2026-11-06 setup run](../2026-11-06_qwen3-4b_extraction_smoke/README.md) with the head ranking fixed. It extracts one function vector (FV) per fictional person but does **not** inject them yet, so it says nothing about whether an FV can make the model answer.

## Reproduce

- **Code:** commit `e20deb4ece18581b6c01cb279e0a2385a55cb5fe` on branch `qwen-fv-extraction`
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (setup-only, same as the 6 Nov run)
- **Command**, from `src/`:

  ```bash
  python extract_knowledge_fv.py --effect logprob --corruption other_person --n_ie_trials 4 \
      --save_path ../results/knowledge_smoke_logprob_otherperson
  ```

  The other defaults were `--model_name Qwen/Qwen3-4B --n_mean_trials 10 --n_top_heads 10 --seed 42`.
- **Environment:** AMD RX 9070 XT (16 GB), ROCm 7.2.1 under WSL2 (kernel 6.18.58.1 after an automatic WSL update), torch 2.9.1, transformers 4.57.6. The model was loaded in float16.
- **Runtime:** about 24 minutes, almost all of it for indirect effects (about 58 s per corrupted prompt, 24 prompts).
- **Interruption:** a first attempt stopped at person 5 of 6 when Windows Update installed WSL 3.0.5.0 and restarted the Linux VM. Its log is `outputs/run_interrupted_by_wsl_update.log`; the results here come from a complete rerun.

## What changed from the 6 Nov run

| | 6 Nov | This run |
|---|---|---|
| Indirect effect | Change in probability of the answer | Change in float32 log-probability of the answer |
| Corrupted prompt | Same person's 4 answers shuffled | The 4 facts of a person who shares none of the answers |
| Corrupted prompts per person | 2 | 4 |

Shuffling usually states a wrong answer to the question being asked, and about a quarter of the time it leaves the queried answer in place. Other-person corruption removes the answer without contradicting it (see `findings/ideas.md`).

A run with default settings reproduces the 6 Nov indirect effects exactly, so the paper's behaviour is unchanged. Mean activations differ slightly from 6 Nov for people after the first, because the extra corrupted prompts change the random-number sequence.

## Results

**The model uses the context.** Same as before: with the facts in context it answered 23 of 24 held-out questions, without context 0 of 24.

**The head ranking is now stable.** Both proposed checks pass:

| Check | 6 Nov | This run | Threshold |
|---|---|---|---|
| Largest single-prompt share of the top-10 effect | 0.99 | 0.07 | Below about 0.30 |
| Split-half top-10 overlap (chance 0.09) | not computed | 7 of 10 | At least 5 |
| Per-person top-10 overlap | 0–5 | 5–9 | |
| Leave-one-person-out overlap with full top 10 | | 9–10 | |
| Distinct effect values per prompt (of 1152 heads) | 1–353 | 1140–1147 | |

- Every prompt contributes 1–7% of the top-10 effect.
- Every person benefits from the shared top-10 heads: patching one of them raises the answer's log-probability by 0.34–0.54 on average for each person.
- Top heads: (31, 3), (32, 8), (24, 26), (34, 1), (33, 29), (31, 2), (35, 25), (29, 11), (26, 20), (29, 24), as (layer, head). Head (31, 3) raises the answer's log-probability by 1.05 on average.
- 5 of these 10 also appeared in the 6 Nov top 10.
- All top heads are in layers 24–35. Positive effect is near zero before layer 14 and peaks at layers 24, 29 and 31–34.
- Some heads lower the answer's probability when patched (most negative average: −0.45).

**FVs still follow the answers rather than the bindings.**

| FV pair | 6 Nov | This run |
|---|---|---|
| Elara and Davor (same cities) | 0.966 | 0.966 |
| Tomas and Mira (same cities) | 0.901 | 0.860 |
| Ines and Joren (same cities) | 0.885 | 0.945 |
| All other pairs | 0.36–0.49 | 0.33–0.47 |

## Interpretation

- The 6 Nov ranking problem was the metric and the corruption, not the method: with log-probability and other-person corruption, the ranking is consistent across prompts and people.
- With a stable shared head set, FVs of people with the same cities in swapped roles are still nearly identical. This strengthens the earlier reading that FVs carry which answers are involved but little about which fact each answer belongs to. A cosine of 0.86–0.97 still doesn't rule out binding; only injection can test that.
- The heads sit in late layers, where the model is about to output the answer. They may mainly push answer tokens rather than carry facts the model can use.
- The FV is meant to be added at a single earlier layer (around 12 by the paper's rule), well before where these heads sit, so the injection layer needs to be swept, including late layers.

## What we investigate next

1. **Injection:** add each person's FV to bare questions (`Q: In which city was Elara Voss born?\nA:`) and check whether the answer appears. Sweep the injection layer, including late layers, and the FV scale.
2. **Controls:** question only, a same-norm random vector, and another person's FV.
3. **Binding:** for same-city pairs, check whether each FV gives the right city for each question, or the same city for both.

Further ideas, such as zero-shot corruption and per-person head selection, are in `findings/ideas.md`.

## Files

- `outputs/`: `summary.json`, `run.log` and `run_interrupted_by_wsl_update.log`. `fv_extraction.pt` (138 MB) is kept locally but not committed, because `*.pt` is git-ignored and GitHub rejects files over 100 MB. Its SHA-256 is `85ec83177065d90aab4efb0d0dccbc5a95b965047c6bd331757b02340ffae23f`.
- `analysis/compare_runs.py`: all numbers above that compare with the 6 Nov run. The 6 Nov folder's `analysis/` scripts also work on this run's outputs.
