#!/bin/bash
#SBATCH --job-name=nntp-clean
#SBATCH --output=runtime/logs/%x-%j.out
#SBATCH --error=runtime/logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=aind

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivan.y.gao@alleninstitute.org

set -e

echo "SLURM job started at: $(date)"

# ========== 1. Move to project root ==========
cd /allen/aind/scratch/ivan.y.gao/CoSyn-RNN

echo "Running on node: $(hostname)"

# TARGET_DIR="figures/experiments-ln"
# TARGET_DIR="figures/similarity-ln"
TARGET_DIR="figures/full-ln"

find "$TARGET_DIR" -type f -print0 | xargs -0 -n 1000 -P 64 rm -f
find "$TARGET_DIR" -type d -empty -delete
rm -rf "$TARGET_DIR"