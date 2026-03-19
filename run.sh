#!/bin/bash
#SBATCH --job-name=T_sweep
#SBATCH --output=T_sweep_%a.log
#SBATCH --array=0-5
#SBATCH --partition=submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

T_VALUES=(25 50 100 250 500 1000)
T=${T_VALUES[$SLURM_ARRAY_TASK_ID]}

source /work/submit/anton100/msci-project/venv/bin/activate
cd /work/submit/anton100/msci-project/FCC-BB-GenAI

SCRIPT=particle_diffusion_snr.py
DATA=guineapig_raw_trimmed.npy
OUTDIR=/work/submit/anton100/msci-project/FCC-BB-GenAI/T_sweep_snr/T_${T}

echo "=== T=${T} (array task ${SLURM_ARRAY_TASK_ID}) ==="

echo "--- TRAINING ---"
python $SCRIPT train \
    --data_path $DATA \
    --outdir $OUTDIR \
    --T $T \
    --epochs 70

echo "--- SAMPLING ---"
python $SCRIPT sample \
    --outdir $OUTDIR \
    --n_events 10000 \
    --sample_batch_size 16

echo "--- EVALUATING ---"
python $SCRIPT evaluate \
    --real_path $DATA \
    --gen_path $OUTDIR/generated_events.npy \
    --outdir $OUTDIR/plots

echo "=== DONE T=${T} ==="