#!/bin/bash
THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
PROJECT="$THIS_SCRIPT_DIR"
APPTAINER="$PROJECT/apptainer/dynamic_loss.sif"

cd "$PROJECT" || exit 1
module load apptainer/1.0 cuda/11.7
echo "Running $1 with container $APPTAINER:"
apptainer run --bind "$(readlink -f "$PROJECT")" --nv --app python "$APPTAINER" "$@"
