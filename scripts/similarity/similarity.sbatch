#!/bin/bash
#SBATCH --job-name=nntp-similarity
#SBATCH --output=runtime/logs/%x-%A-%a.out
#SBATCH --error=runtime/logs/%x-%A-%a.err

#SBATCH --array=0-239

#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --partition=aind
#SBATCH --constraint="h200|a100"
#SBATCH --qos=medium

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ivanygao@outlook.com

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

# ========== 4. Map array index to task pair ==========
IDX="${SLURM_ARRAY_TASK_ID}"
PAIR_INDEX=0
TASK1=""
TASK2=""

N=${#TASKS[@]}
EXPECTED_PAIRS=$((N * (N - 1)))

if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${EXPECTED_PAIRS}" ]]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${IDX}; expected 0 to $((EXPECTED_PAIRS - 1))"
    exit 1
fi

for ((i = 0; i < N; i++)); do
    for ((j = 0; j < N; j++)); do
        if [[ "$i" -eq "$j" ]]; then
            continue
        fi

        if [[ "${PAIR_INDEX}" -eq "${IDX}" ]]; then
            TASK1="${TASKS[$i]}"
            TASK2="${TASKS[$j]}"
            break 2
        fi

        PAIR_INDEX=$((PAIR_INDEX + 1))
    done
done

if [[ -z "${TASK1}" || -z "${TASK2}" ]]; then
    echo "Failed to map SLURM_ARRAY_TASK_ID=${IDX} to a task pair"
    exit 1
fi

# ========== 5. Build Hydra overrides ==========
CONFIG=scripts/similarity/config.yaml

NAME="similarity_${TASK1}_${TASK2}"

TASK1_LIST="[${TASK1}]"
TASK2_LIST="[${TASK2}]"
ALL_TASKS="[${TASK1},${TASK2}]"

echo "Running config: ${CONFIG}"
echo "NAME=${NAME}"
echo "TASK1=${TASK1}"
echo "TASK2=${TASK2}"
echo "TASK1_LIST=${TASK1_LIST}"
echo "TASK2_LIST=${TASK2_LIST}"
echo "ALL_TASKS=${ALL_TASKS}"

# ========== 6. Run ==========
nntp --entrypoint=./src/entrypoint.py run --config="${CONFIG}" \
    "name=${NAME}" \
    "workflow.0.training_tasks=${TASK1_LIST}" \
    "workflow.0.validation_tasks=${ALL_TASKS}" \
    "workflow.0.test_tasks=${ALL_TASKS}" \
    "workflow.1.training_tasks=${TASK2_LIST}" \
    "workflow.1.validation_tasks=${ALL_TASKS}" \
    "workflow.1.test_tasks=${ALL_TASKS}"