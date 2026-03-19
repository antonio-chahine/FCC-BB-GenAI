#!/bin/bash
#SBATCH --job-name=classifier_tsweep
#SBATCH --output=logs/classifier_tsweep_%j.out
#SBATCH --error=logs/classifier_tsweep_%j.err
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

echo "=============================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "=============================="

source /work/submit/anton100/msci-project/venv/bin/activate
cd /work/submit/anton100/msci-project/FCC-BB-GenAI
mkdir -p logs

SWEEP=T_sweep_cosine_charge

# Step 1: prepare RF data
echo ""
echo ">>> Preparing classifier data..."
python classifier_tsweep_combined.py --data \
    --sweep-dir $SWEEP

# Step 2: run RF (CPU, uses all cores)
echo ""
echo ">>> Running RF classifiers..."
python classifier_tsweep_combined.py --run-rf \
    --sweep-dir $SWEEP \
    --n-runs-rf 10

# Step 3: run GNN (GPU)
echo ""
echo ">>> Running GNN classifiers..."
python classifier_tsweep_combined.py --run-gnn \
    --sweep-dir $SWEEP \
    --max-events 5000 \
    --epochs 15 \
    --hidden 32

# Step 4: plot combined results
echo ""
echo ">>> Plotting combined results..."
python classifier_tsweep_combined.py --plot \
    --sweep-dir $SWEEP

echo ""
echo "=============================="
echo "Finished: $(date)"
echo "=============================="