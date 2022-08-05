NUM_GPUS=8

torchrun \
    --nproc_per_node $NUM_GPUS \
    train.py \
    --config-file "e2e_mask_rcnn_R_50_FPN_1x_local.yaml" \
    --dist "nccl"