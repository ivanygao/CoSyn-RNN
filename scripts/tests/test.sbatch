#!/bin/bash
#SBATCH --job-name=nntp-test
#SBATCH --output=runtime/logs/%x-%j.out
#SBATCH --error=runtime/logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=aind
#SBATCH --constraint="a100"

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivanygao@outlook.com

set -e

echo "SLURM job started at: $(date)"

mkdir -p runtime/logs

# ========== 1. Activate Environment ==========
source ~/.bashrc
conda activate nntp-cuda12

# ========== 2. Move to project root ==========
cd /allen/aind/scratch/ivan.y.gao/CoSyn-RNN

echo "Running on node: $(hostname)"

# ========== 3. Run ==========
# CONFIG=scripts/tests/test-8-mdmdrf.yaml
CONFIG=scripts/tests/test-2.yaml

echo "[$SLURM_JOB_ID] Running config: $CONFIG"

nntp --entrypoint=./src/entrypoint.py run --config="${CONFIG}"