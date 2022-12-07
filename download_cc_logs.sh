#!/usr/bin/env bash
THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

cd "$THIS_SCRIPT_DIR" || exit
rsync -chavz \
  --partial \
  --info=progress2 \
  --no-inc-recursive \
  --include='*/' \
  --exclude='*.ckpt' \
  --prune-empty-dirs \
  'cedar:~/scratch/dynamic_loss/logs' cc_logs && \
echo 'Successfully downloaded logs from Cedar'
