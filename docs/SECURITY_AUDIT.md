# Security audit

## Scope and procedure

The ZIP was extracted into a quarantine directory and scanned before repository construction. The scan covered text files, shell scripts, Python sources, result files, logs, nested archive names, and embedded Git metadata. Secret values are intentionally omitted from this report.

## Findings

The source archive contained:

- an OpenAI API credential hard-coded in ten source locations;
- generic credential assignments in shell/Python files;
- personal absolute home-directory paths;
- usernames and cluster-specific filesystem/job configuration;
- raw logs and embedded `.git` directories that could retain historical sensitive content.

Affected paths containing the exposed OpenAI credential were:

- `batch/11_llm_prompt_variants.sh`;
- `hpc_pull/humor_detection_results/scripts/llm_gpt4o_spanish_only.py`;
- `hpc_pull/humor_detection_results/batch/10_llm_gpt4o_spanish.sh`;
- `hpc_pull/humor_detection_results/comb/scripts/llm_gpt4o_comb_only.py`;
- `hpc_pull/humor_temp/scripts/llm_gpt4o_spanish_only.py`;
- `hpc_pull/humor_temp/batch/10_llm_gpt4o_spanish.sh`;
- `hpc_pull/humor_temp/other_lang/scripts/llm_gpt4o_new_langs.py`;
- `hpc_pull/humor_temp/other_lang/batch/19_llm_gpt4o_new_langs.sh`;
- `hpc_pull/humor_temp/comb/scripts/llm_gpt4o_comb_only.py`;
- `hpc_pull/humor_prompt_variants_results.zip::batch/11_llm_prompt_variants.sh`.

Additional generic credential assignments and absolute home paths occurred across legacy Qwen, GPT-Neo/OPT, transformer, Phi-3, and SLURM files; those files were excluded or rewritten rather than copied.

The actual credential value was never copied into the cleaned repository, documentation, test fixtures, command output, or final deliverable.

## Remediation

- All API authentication now uses `OPENAI_API_KEY` from the environment.
- `HF_TOKEN` is available as an optional environment variable for gated Hugging Face access.
- `.env.example` contains variable names only.
- `.env`, logs, caches, checkpoints, model weights, generated predictions, and credentials are ignored.
- All personal and cluster-specific absolute paths were removed.
- No original `.git` directory or historical object was reused.
- GPT-4o-mini requires an explicit `--authorize-paid-api` flag in addition to a configured environment variable.
- A final repository scan is run by `tests/test_no_secrets.py` and `scripts/validate_repository.py`.

## Required owner action

The repository owner must revoke the exposed OpenAI credential and replace it with a newly issued credential stored outside version control. Cleaning this repository does not remove the credential from other copies, shell histories, backups, cluster storage, or any previously published history.

## Final scan status

The completed repository passed the final automated scan for recognizable OpenAI/Hugging Face tokens, private-key blocks, quoted credential assignments, and personal `/Users/<name>/` or `/home/<name>/` paths.
