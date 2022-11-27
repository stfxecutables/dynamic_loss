#!/usr/bin/env bash

THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

cd "$THIS_SCRIPT_DIR" || exit
rsync -chavz \
  --partial \
  --info=progress2 \
  --no-inc-recursive \
  --include='*.events.out.*' \
  --include='*/' \
  --exclude='*.ckpt' \
  --exclude='*.npy' \
  --prune-empty-dirs \
  'cedar:/scratch/dberger/dynamic_loss/logs' cc_logs && \
echo 'Successfully downloaded app logs from Cedar' && \
cd cedar || exit && \
fd --hidden --no-ignore .tar.gz -x tar -xvf {} -C {//}
