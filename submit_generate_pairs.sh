#!/bin/bash
#SBATCH --job-name=gen_pairs
#SBATCH --output=gen_pairs_%a.log
#SBATCH --array=0-4
#SBATCH --partition=submit-gpu
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

EVENTS_PER_TASK=10000
START_INDEX=$(( SLURM_ARRAY_TASK_ID * EVENTS_PER_TASK ))
TASK_SEED=$(( 1000 + SLURM_ARRAY_TASK_ID ))

MODEL_DIR=/work/submit/anton100/msci-project/FCC-BB-GenAI/T_sweep_cosine_extradata/T_250
PAIRS_DIR=/ceph/submit/data/user/a/anton100/bb_pairs/generated_pairs

source /work/submit/anton100/msci-project/venv/bin/activate
cd /work/submit/anton100/msci-project/FCC-BB-GenAI

mkdir -p "${PAIRS_DIR}"

echo "=== Array task ${SLURM_ARRAY_TASK_ID} ==="
echo "Events: ${EVENTS_PER_TASK}, start index: ${START_INDEX}, seed: ${TASK_SEED}"

python generate_pairs.py \
    --outdir            "${MODEL_DIR}"       \
    --pairsdir          "${PAIRS_DIR}"       \
    --n_events          "${EVENTS_PER_TASK}" \
    --sample_batch_size 32                   \
    --start_index       "${START_INDEX}"     \
    --seed              "${TASK_SEED}"       \
    --sanity_batches    2

echo "=== DONE task ${SLURM_ARRAY_TASK_ID} ==="