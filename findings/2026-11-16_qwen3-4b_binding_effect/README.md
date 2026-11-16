# Head selection by binding effect on Qwen3-4B (2026-11-16)

**Branch `binding-effect`.** Experiments 3–4 showed that extracted FVs push one of the person's cities for every question. All of them used heads chosen by the indirect effect on the right answer, and those heads were all in layers 24–35. Hypothesis: that criterion favours heads that write the answer token. Heads whose contribution only helps *together with the question* would rank low and never enter an FV. So here heads are chosen by a **binding effect**: how much patching a head's mean output raises the right city *relative to the person's other cities*. A head that pushes all of the person's cities alike scores zero.

## Reproduce

- **Code:** commit `416d60ac65cf41d0821d5b0a3a3ca0b9b691c1ce` on branch `binding-effect` (adds `effect='binding'` to `src/compute_indirect_effect.py` and `--effect binding` to `src/extract_knowledge_fv.py`; the paper's defaults are unchanged).
- **Data:** `dataset_files/knowledge/fictional_people_smoke.json` (6 people × 4 facts, setup-only), as in experiments 2–3.
- **Commands:** `outputs/run_binding.sh`, from `src/`:

  ```bash
  python extract_knowledge_fv.py --effect binding --corruption other_person --n_ie_trials 4 --save_path ../results/knowledge_smoke_binding
  python inject_knowledge_fv.py --fv_path ../results/knowledge_smoke_binding/fv_extraction.pt --save_path ../results/knowledge_smoke_binding_injection
  ```

  These are experiment 2's settings apart from `--effect`: 10 mean prompts and 4 corrupted prompts per person, top-10 heads, seed 42. Injection uses experiment 3's defaults: layers 0, 3, …, 33 and 35; scales 0.5–8.
- **Analysis**, from the repo root: `python findings/2026-11-16_qwen3-4b_binding_effect/analysis/compare_binding.py findings/2026-11-16_qwen3-4b_binding_effect/outputs findings/2026-11-12_qwen3-4b_injection_smoke/outputs findings/2026-11-12_qwen3-4b_extraction_logprob_otherperson/outputs` (pandas only).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:** 35 minutes (20:55–21:30): about 23 for extraction, 12 for injection.

## What changed from experiment 2

Binding score of a prompt = log p(right city) − log Σ p(person's other cities), in float32. The binding effect of a head is that score with the head's mean output patched in at the last token, minus the score without patching, on the corrupted prompt. The corrupted prompt has another person's facts as demonstrations. Everything else is unchanged.

## Results

**Sanity check:** with the facts in context 23/24 right, without them 0/24, as in experiment 1.

**The ranking changes, but most top heads are still late:**

| | Top-10 heads (layer, head) |
|---|---|
| Binding effect | (24,26), (32,8), (31,3), (33,14), (28,0), (29,11), (33,12), (20,3), (23,10), (33,20) |
| Log-probability effect (experiment 2) | (31,3), (32,8), (24,26), (34,1), (33,29), (31,2), (35,25), (29,11), (26,20), (29,24) |

- 4 heads are shared. Two new heads are in layers 20–23; the other new ones are in layers 28–33.
- Binding effects are small: the top head's mean is 0.12 (in log-ratio units).

**The binding ranking is unstable:**

| | Largest single-prompt share of the top-10 effect | Split-half top-10 overlap |
|---|---|---|
| Binding effect | 0.27 (several prompts negative) | 2 of 10 |
| Log-probability effect (experiment 2) | 0.07 | 7 of 10 |

**Injection: no improvement.**

| | Binding heads | Log-prob heads (experiment 3) |
|---|---|---|
| Best own-FV cell | 0.29 (layer 6, scale 4) | 0.29 (layer 15, scale 2) |
| Cells above 25%, of 65 | 1 | 1 |
| Own FV, mean over layers and scales ≥ 1 | 0.21 | 0.22 |
| Same-cities partner's FV, same average | 0.22 | 0.23 |
| Different-cities person's FV | 0.01 | 0.01 |
| Random vector | 0.05 | 0.05 |

- At scale 8, the own FV again gives one of the person's cities for every question at every layer (at scale 4, every layer except layer 0, with 92%). It almost always gives the same one for all 4 questions (1.0–1.3 distinct cities per person at scales 4 and 8).
- **The twin test fails as before:** the partner's FV, with the same cities in swapped roles, is as accurate as the own FV.
- FVs of twins are still near-identical: cosine 0.97 (Elara/Davor), 0.85 (Tomas/Mira) and 0.96 (Ines/Joren), against 0.31–0.46 for other pairs.

## Interpretation

- **Choosing heads by binding didn't produce a vector that binds.** Accuracy, the one-city-per-person pattern and the twin tie are the same as with the log-probability ranking.
- **This doesn't settle the hypothesis.**
  - The binding effects are small and the ranking rests on 4 corrupted prompts per person: two halves of the prompts agree on only 2 of the top 10.
  - So the selected heads are partly noise, and the test was weak.
  - Most of the selected heads are still in the late answer-writing layers, so the FV is still dominated by heads that push answers.
- **What it does show:** with this amount of data, a binding criterion doesn't single out a stable set of mid-layer heads. The heads that do score are mostly the same late heads.
- **Limits:** the 6 × 4 setup dataset; 4 corrupted prompts per person (the paper uses 25 and 100 mean prompts); top-10 heads only; no capital or unrelated-question condition in the selection.

## What we investigate next

1. **A fair test of the hypothesis** needs a stable binding ranking: 25 corrupted prompts per person (the paper's number), possibly on the 8 × 8 dataset. With the current code that's about 2.5 hours on the 6 × 4 data (58 s per prompt). It's best done after the planned speed-up of the scripts.
2. **Restrict to mid layers:** take the top binding heads among layers below 24 only, to test whether those heads carry binding without the late answer-pushing heads dominating the sum.

## Files

- `outputs/extraction/summary.json`: sanity check, ranking stability, top heads, FV norms and cosine similarities.
- `outputs/injection/`: `injection_results.csv` (one row per question, layer, scale and condition), `injection_summary_table.csv`, `summary.json`.
- `outputs/run_binding.sh`, `outputs/knowledge_smoke_binding.commit`, and both logs.
- `fv_extraction.pt` and the per-person checkpoints stay on the GPU PC (git-ignored). SHA-256 of `fv_extraction.pt`: `1b99125b759da9acd4e7f3297eeb58fdd1b00237d2fdc5f0f2fffc5a2f6e30f5`.
- `analysis/compare_binding.py`: all tables above, compared with experiments 2 and 3.
