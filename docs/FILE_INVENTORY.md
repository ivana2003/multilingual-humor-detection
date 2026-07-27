# Read-only archive inventory

The original ZIP was inspected in quarantine before any file was copied. The final repository was created from scratch. This inventory groups exact duplicates and related experiment families instead of reproducing three redundant directory trees.

Status vocabulary: **KEEP**, **REFACTOR**, **REFERENCE RESULT**, **OPTIONAL ANALYSIS**, **EXCLUDE**, **DUPLICATE**, **SECRET/UNSAFE**, **UNRESOLVED**.

| Original path or group | Apparent purpose | Paper support | Status | Destination / decision | Reason |
|---|---|---|---|---|---|
| `README.txt`; `hpc_pull/humor_detection_git/README.md`; `hpc_pull/humor_temp/other_lang/README.md` | Working notes and old instructions | Partial, but contains stale datasets, scores, and environment assumptions | EXCLUDE | Replaced by `README.md` and audit documents | Claims were not consistently aligned with the final paper. |
| `.DS_Store`, `__MACOSX/**`, `._*` | macOS metadata | None | EXCLUDE | None | Non-scientific metadata. |
| Every `**/.git/**` under the three HPC trees | Embedded repository histories | None | SECRET/UNSAFE | None; new Git history initialized | May contain deleted secrets and unrelated history. |
| `logs/prompt_variants_445369.out` and raw `*.out`/SLURM logs | Cluster execution logs | Not needed to reproduce reported experiments | EXCLUDE | None | Large, path-specific, and potentially sensitive. |
| `batch/11_llm_prompt_variants.sh` and mirrored GPT batch files | Prompt-variant/API launchers | Prompt variants not reported in final paper | SECRET/UNSAFE | None | Contained a hard-coded API credential and cluster paths. |
| Other `batch/*.sh`, `comb/batch/*.sh`, `other_lang/batch/*.sh`; root `hpc_pull/20_transformers_lang_specific.sh` | HPC launch scripts | Partial | REFACTOR | `slurm/*.sbatch` | Replaced by three parameterized, path-independent launchers. |
| `hpc_pull/humor_prompt_variants_results.zip` | Nested duplicate prompt-variant bundle | Not part of final paper pipeline | DUPLICATE | None | Duplicates already extracted prompt-variant files. |
| `scripts/llm_prompt_variants_5langs.py` and `results/llm_prompt_variants*.csv` | Literal/cultural/simplified prompt search | No; Appendix A.4 uses two fixed prompts | EXCLUDE | Mentioned in `docs/REPRODUCIBILITY_REPORT.md` | CV prompt-variant scores must not be confused with held-out Table 2 results. |
| `scripts/statistical_significance_bootsa.py`; `results/statistical_significance_bootsa*.csv` | BootSA significance testing | No significance testing reported | OPTIONAL ANALYSIS / EXCLUDE | None | Complete enough to inspect, but retaining it would blur the paper pipeline. |
| `hpc_pull/humor_detection_git/scripts/create_splits.py`; mirrored `humor_detection_results/scripts/create_splits.py`, `humor_temp/scripts/create_splits.py` | Legacy ColBERT/Spanish loading and 80/20 split creation | Conflicts with final paper | EXCLUDE | Replaced by `src/humor_detection/splitting.py` and `scripts/prepare_data.py` | Uses ColBERT instead of CoMB, lacks validation split, and contains an unsafe fallback that fabricates Spanish labels. |
| `hpc_pull/humor_detection_git/data/dataset_splits_final.json` | Legacy split IDs | Only ColBERT and Spanish; not paper-complete | REFERENCE RESULT / EXCLUDE | Described in reproducibility report | Cannot be used for the five-dataset 70/10/20 protocol. |
| `traditional_ml_cv_test.py` in all three trees | Legacy LR/SVM experiments | Partial | REFACTOR | `scripts/run_traditional_ml.py`, `configs/traditional_ml.yaml` | Removed arbitrary feature caps/naming and centralized exact paper settings. |
| `comb/scripts/traditional_ml_comb_only.py` in results/temp trees | CoMB-specific traditional ML | Intended EN experiment, but incomplete | REFERENCE RESULT / EXCLUDE | Evidence recorded in `table2_traceability.csv` | Requires missing `data/comb_data.csv` and caps CV/test sample counts. |
| `other_lang/scripts/traditional_ml_new_langs.py`, `traditional_ml_new_langs_COMPLETE.py` | PT/ZH/HI traditional ML variants | Partial | REFACTOR | Unified traditional ML pipeline | Contains multiple unpublished feature variants and person-specific labels. |
| All traditional result CSVs, including `results/traditional_ml_comb.csv`, `traditional_ml_cv_and_test.csv`, and `traditional_ml_new_langs*.csv` | Candidate metrics | Mixed held-out/CV evidence | REFERENCE RESULT | `results/archive_evidence/table2_traceability.csv` only | Raw CSVs are not copied because several rows conflict with Table 2 or use unpublished configurations. |
| `transformers_cv_test.py` in duplicated trees | XLM-R and legacy transformer experiments | Partial | REFACTOR | `scripts/run_transformers.py` | Replaced duplicated scripts with config-selected paper models. |
| `comb/scripts/transformers_comb_only.py` | CoMB XLM-R/DeBERTa | Intended EN experiment, incomplete provenance | REFERENCE RESULT / EXCLUDE | Traceability audit | Missing data/split source; the reported DeBERTa value matches CV mean rather than its test column. |
| `other_lang/scripts/transformers_multilingual_PROPER.py`, `transformers_spanish_multilingual.py` | XLM-R runs | Yes, with legacy implementation details | REFACTOR | Unified transformer runner | Consolidated into one implementation. |
| `other_lang/scripts/transformers_lang_specific.py` | BETO, BERTimbau, Chinese BERT, MuRIL | Yes | REFACTOR | Unified transformer runner and config | Archive scores do not support every published cell exactly. |
| `other_lang/scripts/transformers_new_langs_MINIMAL.py`, `deberta_new_langs.py` | Experimental transformer subsets/DeBERTa across languages | No | EXCLUDE | None | Models/runs not reported in Table 2. |
| Transformer result CSVs in root, `comb/results/`, and `other_lang/results/` | Candidate test/CV evidence | Mixed | REFERENCE RESULT | Traceability audit only | Preserved as findings, not presented as reproduced results. |
| `llm_gpt4o_spanish_only.py`, `llm_gpt4o_comb_only.py`, `llm_gpt4o_new_langs.py` and mirrors | GPT-4o-mini evaluation | Intended model, inconsistent prompts/provenance | SECRET/UNSAFE / REFACTOR | `scripts/run_llms.py` | Several copies contained a credential; rewritten with environment variables, caching, and explicit paid-call authorization. |
| `llm_qwen_llama_cv_test.py`, `llm_qwen_llama_cv_test_FINAL.py`, `llm_all_comb_only.py`, `llm_all_models_new_langs.py` | Qwen plus unpublished Llama runs | Qwen only | REFACTOR | Qwen branch of `scripts/run_llms.py` | Llama code removed; Qwen settings aligned with Appendix A.3. |
| `llm_cv_test.py` | Earlier generic LLM pipeline | Partial/obsolete | EXCLUDE | None | Superseded by paper-specific LLM implementation. |
| GPT/Qwen result CSVs in all trees | Candidate metrics | Mixed held-out/CV and prompt variants | REFERENCE RESULT | Traceability audit only | Several values disagree with Table 2 or cannot be tied to Appendix A.4 prompts. |
| `phi3_cv_test.py`, `llm_gptneo_opt_cv_test.py` and corresponding batches/results | Phi-3, GPT-Neo, OPT experiments | No | EXCLUDE | None | Unpublished preliminary models. |
| Llama branches embedded in Qwen scripts/results | Llama experiments | No | EXCLUDE | None | Not reported in Table 2. |
| Generic BERT/RoBERTa/DeBERTa-on-every-language branches | Preliminary encoder comparisons | No | EXCLUDE | None | Only paper-reported model identities remain primary. |
| `requirements.txt` and `.gitignore` in duplicated trees | Environment/hygiene | Partial | REFACTOR | Root dependency files and `.gitignore` | Dependencies split by core, transformer, Qwen, and OpenAI use. |
| `hpc_pull/humor_detection_results/**` versus `hpc_pull/humor_temp/**` | Near-identical working trees | Duplicate | DUPLICATE | Not copied | Files were compared by path/hash and treated as provenance duplicates. |
| `hpc_pull/humor_detection_git/**` | Earlier ColBERT/Spanish repository snapshot | Mostly obsolete | DUPLICATE / EXCLUDE | Not copied | Does not implement the final five-dataset CoMB design. |
| Raw datasets, model checkpoints, weights, environments, prediction caches | Expected experiment artifacts | Required in principle but absent or inappropriate to publish | UNRESOLVED / EXCLUDE | Download instructions and generated output locations | No raw dataset was present; no missing artifact was fabricated. |

## Duplicate-tree conclusion

The archive is not a single coherent repository snapshot. It is a collection of at least three stages of the project, with mirrored scripts and results under `hpc_pull/humor_detection_git`, `hpc_pull/humor_detection_results`, and `hpc_pull/humor_temp`. The new repository uses no direct directory copy from any of them.
