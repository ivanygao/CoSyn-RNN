#!/bin/bash
#SBATCH --job-name=nntp-zip
#SBATCH --output=runtime/logs/%x-%j.out
#SBATCH --error=runtime/logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=aind
#SBATCH --qos=medium

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivanygao@outlook.com

set -e

echo "SLURM job started at: $(date)"

# ========== 1. Move to project root ==========
cd /allen/aind/scratch/ivan.y.gao/CoSyn-RNN

echo "Running on node: $(hostname)"

# ========== 3. Define input/output ==========
folder="./figures/experiments"
output="./figures/experiments.tar.gz"
# folder="./figures/similarity"
# output="./figures/similarity.tar.gz"
# folder="./figures/full"
# output="./figures/full.tar.gz"
partial_output="${output}.partial"

if [[ ! -d "$folder" ]]; then
    echo "ERROR: Folder does not exist: $folder" >&2
    exit 1
fi

rm -f "$partial_output"

# ========== 4. Compress ==========
echo "Compressing: $folder"
echo "Output:      $output"

tar \
    --exclude='*.png' \
    -cf - "$folder" \
| pigz \
    -p "${SLURM_CPUS_PER_TASK:-64}" \
    -9 \
> "$partial_output"

mv "$partial_output" "$output"

# ========== 5. Validate ==========
pigz -t "$output"

echo
echo "Archive created successfully:"
ls -lh "$output"

sha256sum "$output" > "${output}.sha256"

echo "Checksum saved to: ${output}.sha256"
echo "SLURM job completed at: $(date)"