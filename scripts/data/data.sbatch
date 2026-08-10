#!/bin/bash
#SBATCH --job-name=nntp-generate
#SBATCH --output=runtime/logs/%x-%j.out
#SBATCH --error=runtime/logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=aind

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivan.y.gao@alleninstitute.org

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
bash ./scripts/data/generate_data.sh 1 42 128 128128

echo "SLURM job finished at: $(date)"