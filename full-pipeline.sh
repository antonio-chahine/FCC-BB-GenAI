#!/bin/bash
#SBATCH --job-name=particle_full
#SBATCH --output=full_%j.log
#SBATCH --partition=submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source /work/submit/anton100/msci-project/venv/bin/activate
cd /work/submit/anton100/msci-project/FCC-BB-GenAI

SCRIPT=particle_diffusion_new.py
OUTDIR=/work/submit/anton100/msci-project/FCC-BB-GenAI/new_41
DATA=/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy

echo "=== TRAINING ==="
python $SCRIPT train \
  --data_path $DATA \
  --outdir $OUTDIR \
  --epochs 75 \
  --resume

echo "=== SAMPLING ==="
python $SCRIPT sample \
  --outdir $OUTDIR \
  --n_events 10000

echo "=== EVALUATING ==="
python $SCRIPT evaluate \
  --real_path $DATA \
  --gen_path $OUTDIR/generated_events.npy \
  --outdir $OUTDIR/plots