#!/bin/bash
#SBATCH --job-name=humor-transformer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${DATASET:?Set DATASET to en, es, pt, zh, or hi_en}"
: "${MODEL:?Set MODEL to xlm-r or language-specific}"
mkdir -p logs
python scripts/run_transformers.py --dataset "$DATASET" --model "$MODEL"
