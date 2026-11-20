# Binding effect with the paper's prompt counts, mid-layer heads, and faster training (2026-11-16)

Follow-up to the [binding-effect run](../2026-11-16_qwen3-4b_binding_effect/README.md) (experiment 12). That run selected heads by how much they raise the right city relative to the person's other cities, but its ranking was unstable with only 4 corrupted prompts per person. This experiment has three parts:

1. **A fair test of the binding idea:** binding and log-probability rankings, both with the FV paper's prompt counts (100 mean prompts, 25 corrupted prompts per person).
2. **Mid-layer heads only:** FVs from the top heads below layer 24, so the late answer-writing heads can't dominate the sum.
3. **Speed-ups:** a batched indirect-effect computation, and `--fast` / `--early_stop` for contrast training, checked against experiment 10.

## Reproduce

- **Code** (branch `fast-training`):
  - **`6fb20c9f26e1541df5b087ee974dd4e5df05a3ad`:** `--head_batch_size`, which patches one head per batch row and computes logits only at the last position.
  - **`a01618fb454dc9a4b69f06a360ac8db471b3dde0`:** `--fast` (the frozen layers below the intervention run once per batch) and `--early_stop`, which stops once the training loss improves by less than 0.01 over 30 steps.
  - **`ba7be5654c798f71259b2a52c604471052aa1f47`:** `src/restrict_fv_heads.py`.
- **Data:** `fictional_people_smoke.json` (6 × 4) for parts 1–2; `fictional_people_capacity.json` (8 × 8) for part 3.
- **Commands:**
  - `outputs/run_paper_counts.sh`: `extract_knowledge_fv.py --effect {binding,logprob} --corruption other_person --n_mean_trials 100 --n_ie_trials 25 --head_batch_size 128`, then `inject_knowledge_fv.py`.
  - `outputs/run_midlayers.sh`: `restrict_fv_heads.py --max_layer 24` on both extractions, then injection.
  - `outputs/fast_check/run_fast_check.sh`: five experiment-10 settings with `--fast` (one without and four with `--early_stop`).
- **Analysis**, from the repo root: `python findings/2026-11-16_qwen3-4b_binding_paper_counts/analysis/analyse_paper_counts.py findings/2026-11-16_qwen3-4b_binding_paper_counts/outputs` (pandas only).
- **Environment:** AMD RX 9070 XT, ROCm 7.2.1 under WSL2, torch 2.9.1, transformers 4.57.6, float16.
- **Runtime:**
  - Part 1: 71 minutes for both extractions and injections. With the old one-head-per-pass loop, the indirect effects alone would have taken about 5 hours.
  - Part 2: 24 minutes.
  - Part 3: 26 minutes.

**Check of the batched indirect effect,** on one real corrupted prompt against the old loop: 7.5–9.4 s instead of 56–57 s per prompt. Correlation 0.993–0.999, top-10 heads 10/10 (log-prob, prob) and 9/10 (binding). The largest difference was 1–3% of the largest effect, float16 rounding in batched kernels.

## Results

### 1. Rankings with the paper's prompt counts

| Ranking | Split-half top-10 overlap | Largest single-prompt share | Top-10 heads (layer, head) |
|---|---|---|---|
| Binding, 4 prompts (experiment 12) | 2/10 | 0.27 | |
| Binding, 25 prompts | 4/10 | 0.04 | (31,3), (24,26), (32,7), (31,2), (28,0), (22,10), (24,25), (33,12), (21,19), (29,11) |
| Log-prob, 25 prompts | 9/10 | 0.01 | (31,3), (32,8), (24,26), (34,1), (33,29), (31,2), (29,11), (35,25), (23,10), (26,20) |

- With more prompts, the binding ranking is no longer dominated by single prompts, but only half of it is reproducible.
- The log-prob ranking is nearly the same as experiment 2's (9 of 10 heads).

### 2. Injection

Accuracy is averaged over all layers and scales ≥ 1, with the same test as experiment 3.

| Heads | Own FV | Twin's FV | Different person | Random | Best cell | Cells above 25% (of 65) |
|---|---|---|---|---|---|---|
| Binding, all layers | 0.19 | 0.20 | 0.01 | 0.04 | 0.33 (layer 24, scale 2) | 4 |
| Log-prob, all layers | 0.22 | 0.23 | 0.01 | 0.06 | 0.29 (layer 3, scale 4) | 5 |
| Binding, below layer 24 | 0.03 | 0.03 | 0.03 | 0.02 | 0.12 | 0 |
| Log-prob, below layer 24 | 0.03 | 0.03 | 0.01 | 0.02 | 0.21 | 0 |

- **All layers:** the same one-city pattern as experiments 3, 4 and 12.
  - At scales 4 and 8, 98–100% of predictions are one of the person's cities, with 1.06–1.10 distinct cities per person over 4 questions.
  - The twin test fails: the own FV isn't better than the twin's. Twin FV cosines are 0.94–0.96 (binding) and 0.94–0.98 (log-prob).
  - The best cells (8 of 24 questions) are within chance over 65 cells.
- **Below layer 24:** the FVs do about as well as a random vector, and only 26–27% of their predictions are one of the person's cities at scales 4 and 8. The binding heads chosen there are (22,10), (21,19), (22,11), (20,3), (23,10), (21,2), (21,18), (22,0), (19,3), (22,15).

### 3. Training speed-ups (contrast weight 1, 8 × 8 data)

| Run | Time (experiment 10, approx.) | Time now | Mean steps | Held-out right (exp. 10 → now) | Capitals right | Other people → owner's city | KL, other people |
|---|---|---|---|---|---|---|---|
| Head vectors, `--fast` | ~11 min | 6.2 min | 300 | 0.859 → 0.859 | 0.781 → 0.781 | 0.005 → 0.005 | 0.366 → 0.366 |
| Head vectors, `--fast --early_stop` | ~11 min | 4.6 min | 215 | 0.859 → 0.844 | 0.781 → 0.719 | 0.005 → 0.010 | 0.366 → 0.445 |
| Key–value memory, both | ~13 min | 7.4 min | 269 | 0.750 → 0.781 | 0.875 → 0.906 | 0.010 → 0.021 | 0.145 → 0.173 |
| Heads' 8 layers, both | ~11 min | 3.4 min | 164 | 0.844 → 0.891 | 0.688 → 0.656 | 0.000 → 0.016 | 0.328 → 0.476 |
| Single vector at the name, both | ~14 min | 4.2 min | 108 | 0.594 → 0.547 | 0.875 → 0.875 | 0.000 → 0.000 | 0.033 → 0.048 |

The experiment-10 times are from file timestamps and include model loading and evaluation.

- **`--fast` alone reproduces experiment 10 exactly** on all four metrics.
- **Early stopping** moves held-out accuracy by up to 5 points, about the size of a seed change (experiment 11), and leaves slightly larger side effects (KL 0.17–0.48 instead of 0.15–0.37).

## Interpretation

- **Head selection, by any criterion we tried, doesn't make averaged FVs bind facts to questions.** With a stable ranking from the paper's prompt counts, both rankings give the same one-city-per-person vector.
  - **Mid-layer heads:** without the late heads, the averaged outputs carry no usable information about the person at all.
  - **More data:** it doesn't change the 25% ceiling either. The log-prob ranking with the paper's counts is very stable and still fails.
- **Binding heads exist only weakly.** The binding effects are small, and half of the top 10 changes between halves of the prompts. Most of the heads that score are the same late heads.
- **This closes the extraction line on this data:** neither the corruption, the metric, the amount of data, the averaging nor the choice of heads gets past one pushed city.
- **The speed-ups work:** `--fast` gives identical results about twice as fast, and early stopping makes it 2.4–3.4 times faster overall at the cost of seed-level noise.
- **Limits:** the 6 × 4 setup dataset for parts 1–2; top-10 heads only; the "below 24" cut is one choice.

## What we investigate next

Extraction is closed. The project continues with learned interventions on real documents (wiki2026 dataset, next experiment). The fast training options are used from now on.

## Files

- **`outputs/binding/` and `outputs/logprob/`:** `extraction/summary.json`, `injection/` (results and summary) and logs.
- **`outputs/below24/`:**
  - `fv_*_below24.json` (heads and norms), `injection_*_below24/`, and logs;
  - `.pt` files git-ignored, SHA-256 not recorded, recomputable with `restrict_fv_heads.py`.
- **`outputs/fast_check/`:** `results.csv`, `summary.json` and `run.log` per run, plus `timings.txt`.
- **`outputs/*.commit` and `outputs/*.sh`:** the commits and commands used.
- **Not copied:** `fv_extraction.pt` and the per-person checkpoints stay on the GPU PC (git-ignored).
- **`analysis/analyse_paper_counts.py`:** all tables above.
