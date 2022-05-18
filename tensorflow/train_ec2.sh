#!/usr/bin/env bash

GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    -np $GPU_COUNT -H localhost:$GPU_COUNT \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    python ${BASEDIR}/train.py \
    --config configs/single_node_8gpu.yaml