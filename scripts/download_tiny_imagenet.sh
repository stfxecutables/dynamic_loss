#!/usr/bin/env bash
THIS_SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
ROOT="$(dirname "$THIS_SCRIPT_DIR")"
DATA="$ROOT/data/tiny_imagenet"

mkdir -p "$DATA"
cd "$DATA" || exit 1
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip