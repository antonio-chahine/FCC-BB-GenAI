#!/bin/bash
#SBATCH --job-name=particle_full
#SBATCH --output=full_%j.log
#SBATCH --partition=submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /work/submit/anton100/msci-project/venv/bin/activate
cd /work/submit/anton100/msci-project/FCC-BB-GenAI

OUTDIR=/work/submit/anton100/msci-project/FCC-BB-GenAI/new_28
DATA=/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy

echo "=== TRAINING ==="
python particle_diffusion_lrchange.py train \
    --resume

echo "=== SAMPLING ==="
python particle_diffusion_lrchange.py sample \
    --outdir $OUTDIR \
    --n_events 5000

echo "=== EVALUATING ==="
python particle_diffusion_lrchange.py evaluate \
    --real_path $DATA \
    --gen_path $OUTDIR/generated_events.npy \
    --outdir $OUTDIR/plots