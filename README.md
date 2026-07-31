# Multilingual Humor Detection

This repository contains the code used for experiments in **"From Supervised Learning to LLMs: A multilingual Study of Humor Detection"**.
The project compares different approaches to binary humor detection across five languages and language settings:
- English
- Spanish
- Portuguese
- Chinese
- Hindi-English code-mixed text

The models range from traditional machine learning baselines to fine-tuned Transformer models and prompted large language models. The main evaluation metric is Macro F1.

## Models
The experiments include three main groups of models:
- **Traditional Machine Learning**: TF-IDF with Logistic Regression and Linear SVM
- **Transformer Models**: multilingual XLM-R and language-specific models such as DeBERTa, BETO, BERTimbau, BERT-Chinese and MuRIL
- **Large Language Models**: GPT-4o-mini and Qwen2.5-7B-Instruct in zero-shot and few-shot settings

## Datasets
The datasets are not included in this repository. They need to be downloaded from their original sources-

| Code | Language | Dataset |
| --- | --- | --- |
| `en` | English | [CoMB](https://github.com/Humor-Research/Humor-detection) |
| `es` | Spanish | [chistes-nlp](https://github.com/liopic/chistes-nlp) |
| `pt` | Portuguese | [Recognizing Humor in Portuguese](https://github.com/NLP-CISUC/Recognizing-Humor-in-Portuguese) |
| `zh` | Chinese | [HumorWB](https://github.com/zzyjerry/HumorWB) |
| `hi_en` | Hindi-English | [Humor Detection Corpus](https://github.com/Ankh2295/humor-detection-corpus) |


## Repository Structure

''' text
configs/ experiment settings and dataset metadatata
scripts/ data preparation and experiment scripts
src/humor_detection/ shared utilities for data, splits, prompts and metrics
tests/ some tests
requirements-*.txt extra dependencies for Transformers and LLMs


## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ivana2003/multilingual-humor-detection.git
cd multilingual-humor-detection

python3 -m venv .venv
source .venv/bin/activate
```

Install the basic dependencies:

```bash
pip install pandas numpy scikit-learn pyyaml pytest
```

Then add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"
```

Install the additional dependencies needed for the experiments you want to run:

```bash
# Fine-tuned Transformer models
pip install -r requirements-transformers.txt

# GPT-4o-mini through the OpenAI API
pip install -r requirements-openai.txt

# Local Qwen experiments
pip install -r requirements-qwen.txt
```

## Preparing the data

Each dataset is converted to the same format:

```text
id,text,label
```

Labels use `0` for non-humorous text and `1` for humorous text. The preparation script also creates deterministic stratified train, validation and test splits using a 70/10/20 ratio.

Example:

```bash
python scripts/prepare_data.py \
  --dataset en \
  --input path/to/dataset.csv \
  --text-column text \
  --label-column label
```

The available dataset codes are `en`, `es`, `pt`, `zh` and `hi_en`.

Use `--id-column` when the source already contains stable identifiers, and `--separator` when a file uses a separator other than a comma.

For the Spanish dataset, ratings are converted to binary labels automatically: rating `1` becomes non-humorous, while ratings `2` and `3` become humorous.

## Running the experiments

### Traditional machine learning

```bash
python scripts/run_traditional_ml.py --dataset en --model lr
python scripts/run_traditional_ml.py --dataset en --model svm
```


### Transformer models

Run XLM-R:

```bash
python scripts/run_transformers.py --dataset en --model xlm-r
```

Run the language-specific model configured for a dataset:

```bash
python scripts/run_transformers.py --dataset en --model language-specific
```


### GPT-4o-mini

Set the API key first:

```bash
export OPENAI_API_KEY="your-key"
```

Then run a zero-shot or few-shot experiment:

```bash
python scripts/run_llms.py \
  --dataset en \
  --backend openai \
  --shot zero \
  --authorize-paid-api
```


### Qwen

```bash
python scripts/run_llms.py \
  --dataset en \
  --backend qwen \
  --shot few
```


## Outputs

Each completed experiment saves:

- predictions for the held-out test set;
- macro F1, accuracy, confusion matrix and classification report;
- the main model and run settings;
- information about the split used for the experiment.

Results are stored under:

```text
results/reproduced/<model-family>/<dataset>/<model>/
```

## Tests

Run the test suite with:

```bash
PYTHONPATH=src pytest
```

The tests cover label conversion, dataset validation, deterministic splitting, prompts, output parsing, metrics and result aggregation.

