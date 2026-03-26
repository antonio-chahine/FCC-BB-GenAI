#!/bin/bash
#SBATCH --job-name=hyperparam_tune
#SBATCH --output=tune_%j.log
#SBATCH --partition=submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source /work/submit/anton100/msci-project/venv/bin/activate

cd /work/submit/anton100/msci-project/FCC-BB-GenAI

python hyperparam_tune_sinh_zplane.py \
    --data_path /work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy \
    --outdir /work/submit/anton100/msci-project/FCC-BB-GenAI/tuning_runs_sinh_zplane_v2 \
    --n_trials 100 \
    --study_name particle_diff_sinh_zplane_v2 \
    --storage sqlite:////work/submit/anton100/msci-project/FCC-BB-GenAI/tuning_sinh_zplane_v2.db