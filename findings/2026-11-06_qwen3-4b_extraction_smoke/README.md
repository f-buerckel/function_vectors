# FV extraction setup run on Qwen3-4B (2026-11-06)

A setup run to check that the function vector (FV) pipeline works for document knowledge on Qwen3-4B. It extracts one FV per fictional person but does **not** inject them yet, so it says nothing about whether an FV can make the model answer.

## Reproduce

- **Code:** commit `31d6170b164d85cd61f7585966a4d99e07a16910` on branch `qwen-fv-extraction`
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (setup-only, not the research dataset)
- **Command**, from `src/`:

  ```bash
  python extract_knowledge_fv.py --save_path ../results/knowledge_smoke
  ```

  The defaults were `--model_name Qwen/Qwen3-4B --n_mean_trials 10 --n_ie_trials 2 --n_top_heads 10 --seed 42`.
- **Environment:** AMD RX 9070 XT (16 GB), ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6. The model was loaded in float16.
- **Runtime:** about 15 minutes, almost all of it for indirect effects (about 59 s per corrupted prompt, 1152 heads).

## Setup

- 6 fictional people with 4 city facts each: born in, works in, studied in and lives in.
- Clean prompt: all 4 facts as `Q:`/`A:` demonstrations, then a paraphrase of one question.
- Corrupted prompt: the repo's default, with the 4 answers shuffled.
- Indirect effect: the repo's default, the change in probability of the answer's first token. All answers are single tokens.
- Heads were ranked across all 6 people, and one FV was built per person from the top 10.
- The dataset happens to contain 3 pairs of people with the same 4 cities in swapped roles: Elara Voss and Davor Quill, Tomas Wrenfield and Mira Solbrandt, Ines Kallow and Joren Thale.

## Results

**The model uses the context.** With the facts in context it answered 23 of 24 held-out questions; without context it answered 0 of 24.

**The head ranking is not usable.**
- One of the 12 corrupted prompts produced 99% of the top-10 heads' total effect.
- One prompt had an effect of exactly 0 for all 1152 heads, and 7 of 12 peaked below 0.0001.
- Many values are multiples of 5.96e-8, the smallest float16 step, and some prompts have only 1–19 distinct values across all heads.
- Top-10 heads agree poorly between people (0–5 of 10 shared).
- The likely cause is that raw probability barely changes when an answer starts near 0 or near 1. The starting probabilities were not logged, so this is an inference.

**FVs differ by person, but follow the answers rather than the bindings.**

| FV pair | Cosine similarity |
|---|---|
| Elara and Davor (same cities) | 0.966 |
| Tomas and Mira (same cities) | 0.901 |
| Ines and Joren (same cities) | 0.885 |
| All other pairs | 0.36–0.49 |

The same pattern appears in head activations without the selected heads:

| Activations compared | Same cities | Different cities |
|---|---|---|
| Layer 12, all heads | 0.981 | 0.975 |
| Layer 24, all heads | 0.948 | 0.892 |
| Layer 33, all heads | 0.914 | 0.395 |
| 10 random heads, layers 8–16 | 0.973 | 0.960 |
| 10 random heads, layers 21–35 | 0.939 | 0.752 |
| The 10 selected heads | 0.913 | 0.470 |

## Interpretation

- Early and middle layers are nearly identical across people, so a shared signal dominates there, probably the QA task (untested).
- Differences between people appear in late layers and track which answers are involved. All 10 selected heads are in layers 22–33.
- So FVs carry content, not only the task. But swapping which fact goes with which city barely changes them, which is what a bag of answers would look like.
- Limits: only 3 same-city pairs, a cosine of 0.97 does not rule out binding, and the selected heads come from an unreliable ranking. Only injection can settle this.

## What we investigate next

Fix the head ranking before injecting, so a failed injection can't just mean bad heads:

1. **Log-probability effect** on float32 logits, so changes to answers near 0 still register.
2. **Other-person corruption:** demonstrations from a person with none of the same answers, so the queried answer is absent from the corrupted prompt.
3. **4 corrupted prompts per person** instead of 2.

Proceed to injection if both of these hold (thresholds are proposals):
- No single prompt contributes more than about 30% of the top-10 effect.
- Two disjoint halves of the people share at least 5 of their top 10 heads (chance is about 0.09).

## Files

- `outputs/`: `summary.json` and `run.log` from the run. `fv_extraction.pt` (138 MB) is kept locally but not committed, because `*.pt` is git-ignored and GitHub rejects files over 100 MB. Its SHA-256 is `c8dd807bb81635d432456e494812271677df6364ed85ddb8da3724ad066f0922`.
- `analysis/inspect_indirect_effect.py`: indirect-effect checks.
- `analysis/compare_activation_similarity.py`: the activation similarity tables.

Run both from the `function_vectors` root with this folder's `outputs` path as the argument.
