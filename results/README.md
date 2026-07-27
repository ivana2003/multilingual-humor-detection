# Results

- `paper/table2_reported.csv`: values transcribed from Table 2. They are published reference values, not newly reproduced outputs.
- `reproduced/`: generated experiment results. Each completed run stores `result.json` and prediction-level JSONL.
- `archive_evidence/table2_traceability.csv`: audit linking published cells to candidate files in the supplied raw archive.

`aggregate_results.py` accepts only results explicitly marked `split: held_out_test` and reads `metrics.test.macro_f1`. Cross-validation means, validation metrics, prompt-variant scores, and debug subsets are excluded.
