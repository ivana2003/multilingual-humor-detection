# Validation report

Validation was limited to safe local checks. No raw dataset was downloaded, no transformer or Qwen checkpoint was downloaded, no GPU training was launched, and no OpenAI request was made.

## Archive audit

- 497 archive entries inventoried.
- 127 non-macOS, non-Git files assessed individually or as documented duplicate groups.
- 369 files inside embedded Git metadata excluded.
- 44 files triggered at least one initial security rule, including credential assignments, personal absolute paths, or the exposed API credential.
- The original ZIP and PDF were not modified or copied into the repository.

## Completed checks

| Check | Outcome |
|---|---|
| Python parsing and byte-compilation for `src/`, `scripts/`, and `tests/` | Passed |
| Unit test suite | 17 tests passed |
| Deterministic 70/10/20 splits, disjointness, coverage, and stratification | Passed |
| Spanish rating mapping and rejection of invalid labels | Passed |
| Expected dataset-size and class-distribution validation | Passed |
| Exact Appendix A.4 prompt snapshots | Passed |
| Strict binary parser and invalid-output fallback to label 0 | Passed |
| Exactly five training examples per class for few-shot prompts | Passed |
| Actual traditional-ML runner on a synthetic dataset | Passed |
| Aggregator selection of held-out test macro-F1 and exclusion of debug results | Passed |
| Parsing of four YAML configurations, `pyproject.toml`, and `CITATION.cff` | Passed |
| `--help` for all six command-line scripts | Passed |
| Import of package and all entry-point modules without starting experiments | Passed |
| Editable package build using locally installed build tooling | Passed |
| Final secret, private-key, credential-assignment, and personal-path scan | Passed |
| Scan for excluded model references in primary code/configuration | Passed |
| Reported/reproduced result-directory separation and `.gitignore` rules | Passed |

## Local core environment

The safe checks were executed with:

- Python 3.13.5;
- scikit-learn 1.8.0;
- pandas 2.2.3;
- NumPy 2.3.5;
- PyYAML 6.0.3.

The project metadata supports Python 3.11–3.13 and bounds core dependencies. GPU and OpenAI optional dependencies were not installed or exercised as part of repository construction.

## Deliberately not validated

The following require missing data, external services, or substantial hardware and were therefore not claimed as successful:

- rebuilding any of the five real datasets;
- reproducing any Table 2 score;
- fine-tuning XLM-R or a language-specific transformer;
- running Qwen 2.5 7B Instruct;
- calling GPT-4o-mini;
- confirming historical few-shot example identities;
- validating dataset redistribution permissions.
