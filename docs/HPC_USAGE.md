# HPC usage

The SLURM launchers contain no usernames, home directories, account names, partitions, or cluster modules. Adapt resource directives to the target cluster without committing private infrastructure details.

## Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the dependency group required by the job. Keep model caches and datasets in locations allowed by the cluster's policy; provide those locations through environment variables rather than editing scientific code.

## Traditional ML

```bash
DATASET=pt MODEL=svm sbatch slurm/traditional_ml.sbatch
```

## Transformers

```bash
DATASET=es MODEL=xlm-r sbatch slurm/transformers.sbatch
DATASET=es MODEL=language-specific sbatch slurm/transformers.sbatch
```

## Qwen

```bash
DATASET=zh SHOT=zero sbatch slurm/qwen.sbatch
```

## GPT-4o-mini

No batch file is supplied for paid API calls. Run it interactively or through a site-approved secret injection mechanism only after reviewing the request estimate. Do not place credentials in an SBATCH file, command history, log, or source code.

## Output and resumption

Generated files go under ignored `results/reproduced/` paths. Qwen and OpenAI predictions are appended as JSONL and reused by prompt/model hash, allowing interrupted jobs to resume without repeating completed requests.
