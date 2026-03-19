#!/bin/bash
#SBATCH --job-name=gnn_classifier
#SBATCH --output=logs/gnn_%j.out
#SBATCH --error=logs/gnn_%j.err
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

source /work/submit/anton100/msci-project/venv/bin/activate
cd /work/submit/anton100/msci-project/FCC-BB-GenAI

T_DIR=T_sweep_cosine_charge/T_500

# ── Step 1: build graph cache ──────────────────────────────────────────────
echo ""
echo ">>> Building graph cache..."
python gnn_classifier.py --data \
    --gen-subdir $T_DIR \
    --max-events 5000 \
    --k 8

# ── Step 2: run classifier + ablation ─────────────────────────────────────
echo ""
echo ">>> Running GNN classifier + feature ablation..."
python gnn_classifier.py --run \
    --gen-subdir $T_DIR \
    --epochs 15 \
    --batch-size 64 \
    --hidden 32 \
    --lr 1e-3