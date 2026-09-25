#!/bin/bash
#SBATCH --job-name=nntp-sequence
#SBATCH --output=/allen/aind/scratch/denis.turcu/CoSyn-RNN/scripts/sequence/runtime/logs/%x-%j.out
#SBATCH --error=/allen/aind/scratch/denis.turcu/CoSyn-RNN/scripts/sequence/runtime/logs/%x-%j.err

#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=aind
#SBATCH --constraint="h200"
#SBATCH --qos=medium

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=denis.turcu@alleninstitute.org

set -Eeuo pipefail
trap 'status=$?; echo "ERROR: line $LINENO: $BASH_COMMAND (exit $status)" >&2; exit "$status"' ERR

echo "SLURM job started at: $(date)"

readonly PROJECT_ROOT=/allen/aind/scratch/denis.turcu/CoSyn-RNN
readonly MICROMAMBA=/home/denis.turcu/.local/bin/micromamba
readonly CONFIG_NAME=${1:?"Usage: sbatch sequence-dt.sbatch <config.yaml>"}
readonly CONFIG="$PROJECT_ROOT/scripts/sequence/$CONFIG_NAME"

if [[ ! -f "$CONFIG" ]]; then
	echo "Config file not found: $CONFIG" >&2
	exit 2
fi

cd "$PROJECT_ROOT"

echo "Running on node: $(hostname)"

echo "[$SLURM_JOB_ID] Running config: $CONFIG"

"$MICROMAMBA" run -n nntp-cuda13 \
	nntp --entrypoint=./src/entrypoint.py run --config="$CONFIG"

# sbatch sequence-dt.sbatch frdmmd.yaml
# sbatch sequence-dt.sbatch mddmrf.yaml