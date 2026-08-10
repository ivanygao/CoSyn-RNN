#!/bin/bash
#SBATCH --job-name=nntp-baseline
#SBATCH --output=runtime/logs/%x-%A-%a.out
#SBATCH --error=runtime/logs/%x-%A-%a.err

#SBATCH --array=0-15

#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=aind
#SBATCH --constraint="h200|a100"
#SBATCH --qos=medium

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivan.y.gao@alleninstitute.org

set -e

echo "SLURM job started at: $(date)"

# 注意：runtime/logs 最好在 sbatch 提交前就存在
mkdir -p runtime/logs

# ========== 1. Activate Environment ==========
source ~/.bashrc
conda activate nntp-cuda12

# ========== 2. Move to project root ==========
cd /allen/aind/scratch/ivan.y.gao/CoSyn-RNN

echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

# ========== 3. Define task list ==========
TASKS=(
    "fdgo-ry"
    "reactgo-ry"
    "delaygo-ry"
    "fdanti-ry"
    "reactanti-ry"
    "delayanti-ry"
    "dm1-ry"
    "dm2-ry"
    "contextdm1-ry"
    "contextdm2-ry"
    "multidm-ry"
    "delaydm1-ry"
    "delaydm2-ry"
    "contextdelaydm1-ry"
    "contextdelaydm2-ry"
    "multidelaydm-ry"
)

# ========== 4. Map array index to single task ==========
IDX="${SLURM_ARRAY_TASK_ID}"
N=${#TASKS[@]}

if (( IDX < 0 || IDX >= N )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}; valid range is 0-$((N - 1))"
    exit 1
fi

TASK="${TASKS[$IDX]}"

# ========== 5. Build Hydra overrides ==========
CONFIG=scripts/baseline/config.yaml

NAME="baseline_${TASK}_${TASK}"

TASK_LIST="[${TASK}]"

echo "Running config: ${CONFIG}"
echo "NAME=${NAME}"
echo "TASK=${TASK}"
echo "TASK_LIST=${TASK_LIST}"

# ========== 6. Run ==========
nntp --entrypoint=./src/entrypoint.py run --config="${CONFIG}" \
    "name=${NAME}" \
    "workflow.0.training_tasks=${TASK_LIST}" \
    "workflow.0.validation_tasks=${TASK_LIST}" \
    "workflow.0.test_tasks=${TASK_LIST}"