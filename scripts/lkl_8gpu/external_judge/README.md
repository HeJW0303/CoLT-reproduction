# External-judge image evaluation

This pipeline evaluates `MathVista_MINI`, `MathVerse_MINI`, and `MMVet` with
`deepseek-v4-flash` over the DeepSeek OpenAI-compatible Chat Completions API at
`low` reasoning effort with thinking disabled. It uses a
separate result root from the existing exact-matching evaluation:

```text
eval/external_judge/results/
```

The result fingerprint includes the inference settings, selected datasets,
judge model, wire protocol, reasoning effort, API concurrency, retry count,
model files, and relevant source code. Re-running the same fingerprint resumes
its prediction and judge checkpoints; a different judge configuration cannot
reuse them. A source-code change also intentionally creates a new result
directory, rather than mixing results produced by different evaluation code.

Each external-judge run has two phases: distributed GPU inference first, then
single-process judge evaluation after every inference rank has exited. This
prevents a long-running judge from timing out an idle distributed rank. On a
resume, records with terminal API-failure markers are retried while successful
checkpoint records are retained. Any completed judge artifacts that need to be
recomputed are archived with a `.failed-judge-*.bak` suffix.

## Run

Activate the `colt` environment, then download and evaluate all three datasets:

```bash
conda activate colt
bash scripts/lkl_8gpu/external_judge/run.sh download
bash scripts/lkl_8gpu/external_judge/run.sh eval official --gpus 4,5,6,7
```

Use `all` to run both stages, or select one or more configured datasets:

```bash
bash scripts/lkl_8gpu/external_judge/run.sh all codefaithful --gpus 4,5,6,7
bash scripts/lkl_8gpu/external_judge/run.sh eval official \
  --datasets MathVista_MINI,MMVet --gpus 4,5,6,7
```

API concurrency defaults to 8. MathVerse performs two judge stages.
It can be changed without affecting GPU inference concurrency:

```bash
bash scripts/lkl_8gpu/external_judge/run.sh eval official --api-nproc 2
```

`--dry-run` prints the resolved non-secret environment and command without
reading the auth file, starting inference, or making an API request.

## Credentials

At execution time, `load_codex_api.py` reads `DEEPSEEK_API_KEY` and injects it
only into the child process as `OPENAI_API_KEY`; it sets the API endpoint to
`https://api.deepseek.com/chat/completions` and never prints the key. Do not
copy credentials into the repository or `.env`.

Set the key in the shell environment before launching. The optional legacy
Codex provider remains available through `--api-provider codex` and then reads
the configured `--codex-config` and `--codex-auth` files.

## Validation

Validation runs automatically after evaluation. It checks source and prediction
row counts, unique and exact index sets, nonempty predictions, every judge
checkpoint and intermediate table, failed judge records, final score files,
MathVista/MathVerse accuracy, and MMVet's continuous mean score.

## Five-model suite

The checked-in five-model launcher evaluates the four checkpoints under
`checkpoints/` and the local Qwen3-VL-8B-Instruct baseline serially:

```bash
bash scripts/lkl_8gpu/external_judge/run_five_models.sh --dry-run
bash scripts/lkl_8gpu/external_judge/run_five_models.sh
```

It verifies all datasets and all five models before the first evaluation. Each
model has an isolated result and log root under `external_judge/five_models`.
The launcher activates `/data/nvme0/lkl/conda/envs/colt` itself. Re-running the
command resumes matching prediction and judge checkpoints. Its default judge
API concurrency is 8; pass `--api-nproc N` to override it.

## Oracle-K fixed-K sweep

To compare the same Oracle-K checkpoint at fixed K values, run the external
judge sweep. It evaluates `K=1..8` serially on all three external-judge
datasets. Each K uses an independent result root, logs the predictor output for
audit, and is validated before the next K begins:

```bash
bash scripts/lkl_8gpu/external_judge/run_oracle_k_fixed_k_sweep.sh --dry-run
bash scripts/lkl_8gpu/external_judge/run_oracle_k_fixed_k_sweep.sh
```

The sweep uses the five-model suite's `training-consistent` latent-transition
protocol. It is intentionally separate from the older `ChartQA/TextVQA`
fixed-K integration sweep, which uses the official transition protocol.
