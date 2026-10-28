#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 CONFIG_JSON" >&2
    exit 2
fi

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$source_root"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python_bin="${NEURALPOM_PYTHON:-python}"
workers="${NEURALPOM_NPROC:-8}"
config="$1"

for action in prepare pretrain midtrain4 posttrain; do
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') starting $action" >&2
    if [[ "$action" == prepare ]]; then
        "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$workers" \
            -m neuralpom.v1.cli --config "$config" prepare
    else
        "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$workers" \
            -m neuralpom.v1.cli --config "$config" train --stage "$action"
    fi
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') completed $action" >&2
done
