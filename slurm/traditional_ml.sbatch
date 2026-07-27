#!/bin/bash
#SBATCH --job-name=humor-ml
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${DATASET:?Set DATASET to en, es, pt, zh, or hi_en}"
: "${MODEL:?Set MODEL to lr or svm}"
mkdir -p logs
python scripts/run_traditional_ml.py --dataset "$DATASET" --model "$MODEL"
