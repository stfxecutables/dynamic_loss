#!/bin/bash
#SBATCH --account=rrg-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --job-name=base_train
#SBATCH --output="/scratch/dberger/dynamic_loss/slurm_logs/base_train__%A_%a_%j.out"
#SBATCH --array=0-149
#SBATCH --time=00-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=31000M

SCRATCH="$(readlink -f "$SCRATCH")"
PROJECT="$SCRATCH/dynamic_loss"
RUN_SCRIPT="$PROJECT/run_python.sh"

PY_SCRIPTS="$PROJECT/scripts"
PY_SCRIPT="$(readlink -f "$PY_SCRIPTS/base_train.py")"

bash "$RUN_SCRIPT" "$PY_SCRIPT"