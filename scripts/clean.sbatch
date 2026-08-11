#!/bin/bash
#SBATCH --job-name=nntp-clean
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

# ========== 1. Move to project root ==========
cd /allen/aind/scratch/ivan.y.gao/CoSyn-RNN

echo "Running on node: $(hostname)"

find wandb -type f -print0 | xargs -0 -n 1000 -P 64 rm -f

find wandb -type d -empty -delete

rm -rf wandb