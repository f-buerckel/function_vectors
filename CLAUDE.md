# Research project: function vectors for document knowledge

Fork of the Function Vectors repo (Todd et al., 2024), used to test whether FVs can carry document knowledge. Our work is on branch `qwen-fv-extraction`.

## Documenting experiments

Every experiment is documented twice: in detail in its own folder, and in short in `findings/progress.md`.

**1. Commit the code first.** The experiment's README names the exact commit that produced its results, so the code must be committed before the run (or before writing up, if it already ran from that code). Findings go in a separate, later commit.

**2. Experiment folder:** `findings/YYYY-MM-DD_<model>_<short-name>/`. The name must not end in `results`, because `.gitignore` ignores `*results`.

- `README.md` with these sections:
  - **Reproduce:** full commit hash and branch, data file, exact command, environment (GPU, ROCm, torch, transformers, dtype), runtime, and any interruptions.
  - **Setup** or **What changed:** what this run does differently from the previous experiment, and why.
  - **Results:** tables of the key numbers. Every number must come from the outputs or analysis scripts.
  - **Interpretation:** what the results mean, with their limits.
  - **What we investigate next.**
  - **Files.**
- `outputs/`: `summary.json`, logs and small CSVs. Large `.pt` files stay local (`*.pt` is git-ignored, and GitHub rejects files over 100 MB); record their SHA-256 in the README.
- `analysis/`: scripts that recompute the README's numbers, runnable from the repo root with the outputs folder as argument.

**3. Short entry in `findings/progress.md`**, newest last:
- heading with number, short name and date;
- link to the folder and the short commit hash;
- **What** (first entry) or **Changed from N**, then **Results**, as a few short bullets.

Update the `## Next` section at the end of `progress.md` to match.

**4. Ideas** for later go in `findings/ideas.md`, each with a date and the commit hash current at the time.

Before committing findings, check every number and claim in the README and `progress.md` against the outputs.
