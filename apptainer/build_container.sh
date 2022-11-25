#!/bin/bash
THIS_SCRIPT_PARENT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd "$THIS_SCRIPT_PARENT" || exit 1
PILFER="$THIS_SCRIPT_PARENT/pilfer_container_files.sh"
#
sudo apptainer build dynamic_loss.sif build_centos.def && bash "$PILFER"
