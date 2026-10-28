#!/usr/bin/env bash
set -euo pipefail
python_bin="${NEURALPOM_PYTHON:-python}"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$source_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
# One fixed, separately checked local rendezvous avoids the standalone
# ephemeral TCPStore collision observed at the PastNet stage transition.
exec "$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
    --rdzv_backend=static --master_addr=127.0.0.1 \
    --master_port="${NEURALPOM_AI_MASTER_PORT:-29617}" \
    -m neuralpom.ai_baselines.cli "$@"
