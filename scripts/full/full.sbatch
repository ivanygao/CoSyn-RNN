#!/bin/bash
#SBATCH --job-name=nntp-full
#SBATCH --output=runtime/logs/%x-%A-%a.out
#SBATCH --error=runtime/logs/%x-%A-%a.err

#SBATCH --array=0-3

#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=aind
#SBATCH --constraint="h200|a100"
#SBATCH --qos=medium

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivan.y.gao@alleninstitute.org

set -e

echo "SLURM job started at: $(date)"

# Slurm 会在脚本启动前创建日志文件，
# 因此 runtime/logs 最好在 sbatch 前就存在。
mkdir -p runtime/logs

# ========== 1. Activate environment ==========
source ~/.bashrc
conda activate nntp-cuda12

# ========== 2. Move to project root ==========
cd /allen/aind/scratch/ivan.y.gao/Neural-Network-Training-Pipeline

echo "Running on node: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}"

# ========== 3. Define seeds ==========
SEEDS=(
    1
    42
    128
    128128
)

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#SEEDS[@]} )); then
    echo "Invalid array index: ${ARRAY_INDEX}"
    exit 1
fi

SEED="${SEEDS[$ARRAY_INDEX]}"

# ========== 4. Define final task order ==========
# min
ORDERED_TASKS=(
    "delaydm1-ry"
    "delaydm2-ry"
    "dm1-ry"
    "contextdm1-ry"
    "multidm-ry"
    "contextdm2-ry"
    "dm2-ry"
    "contextdelaydm2-ry"
    "contextdelaydm1-ry"
    "multidelaydm-ry"
    "delaygo-ry"
    "delayanti-ry"
    "fdanti-ry"
    "fdgo-ry"
    "reactgo-ry"
    "reactanti-ry"
)
# max
# ORDERED_TASKS=(
#     "dm2-ry"
#     "fdgo-ry"
#     "contextdm1-ry"
#     "delayanti-ry"
#     "contextdelaydm1-ry"
#     "delaydm2-ry"
#     "multidm-ry"
#     "reactgo-ry"
#     "contextdelaydm2-ry"
#     "delaydm1-ry"
#     "delaygo-ry"
#     "dm1-ry"
#     "reactanti-ry"
#     "multidelaydm-ry"
#     "fdanti-ry"
#     "contextdm2-ry"
# )

echo "Using seed=${SEED}"
echo "Task order:"

for i in "${!ORDERED_TASKS[@]}"; do
    printf "  workflow.%d: %s\n" "$i" "${ORDERED_TASKS[$i]}"
done

# ========== 5. Build all-task list ==========
# Result:
# [contextdm1-ry,dm1-ry,...,reactanti-ry]
ALL_TASKS="$(
    IFS=,
    echo "[${ORDERED_TASKS[*]}]"
)"

echo "ALL_TASKS=${ALL_TASKS}"

# ========== 6. Generate workflow overrides ==========
WORKFLOW_OVERRIDES=()

for i in "${!ORDERED_TASKS[@]}"; do
    TASK="${ORDERED_TASKS[$i]}"

    WORKFLOW_OVERRIDES+=(
        "workflow.${i}.training_tasks=[${TASK}]"
        "workflow.${i}.validation_tasks=${ALL_TASKS}"
        "workflow.${i}.test_tasks=${ALL_TASKS}"
    )
done

# ========== 7. Run ==========
CONFIG="scripts/full/config.yaml"

echo "Running config: ${CONFIG}"
echo "Overriding seed: ${SEED}"

nntp --entrypoint=./src/entrypoint.py run --config="${CONFIG}" \
    "seed=[${SEED}]" \
    "${WORKFLOW_OVERRIDES[@]}"

echo "SLURM job finished at: $(date)"