#!/bin/bash
#SBATCH --job-name=humor-qwen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${DATASET:?Set DATASET to en, es, pt, zh, or hi_en}"
: "${SHOT:?Set SHOT to zero or few}"
mkdir -p logs
python scripts/run_llms.py --dataset "$DATASET" --backend qwen --shot "$SHOT"
