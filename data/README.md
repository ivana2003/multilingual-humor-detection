# Data preparation

This repository does not contain raw datasets. Download each corpus from its official source, check its terms, and normalize it locally with `scripts/prepare_data.py`.

## Canonical normalized schema

Generated files use three columns:

- `id`: a stable, unique record identifier;
- `text`: the input text;
- `label`: integer `0` for not humorous or `1` for humorous.

The preparation script rejects missing text, non-binary labels after mapping, duplicate identifiers, unexpected dataset sizes, and class proportions that do not round to the one-decimal percentages reported in the paper.

## Split manifests

A manifest under `data/splits/<dataset>.json` records:

- seed and 70/10/20 ratios;
- train, validation, and test identifiers;
- split sizes and class counts;
- the normalized source file SHA-256;
- per-split identifier hashes;
- dataset name and schema version.

Splits are stratified and deterministic with seed 42. The code verifies that identifiers are unique and that no item appears in more than one split.

## Spanish labels

The paper's mapping is mandatory:

- original rating 1 -> label 0;
- original rating 2 -> label 1;
- original rating 3 -> label 1.

The Spanish corpus is the Martínez jokes collection, not HAHA/IberLEF 2018.

## Expected statistics

| Dataset | Total | Humorous |
|---|---:|---:|
| CoMB (`en`) | 45,488 | 62.2% |
| Martínez Spanish jokes (`es`) | 2,419 | 63.1% |
| Portuguese (`pt`) | 2,800 | 50.0% |
| HumorWB/Weibo (`zh`) | 5,291 | 38.0% |
| Hindi-English (`hi_en`) | 2,949 | 59.6% |

The percentages in the paper are rounded. Validation accepts a generated fraction only when its percentage rounds to the same one-decimal value.
