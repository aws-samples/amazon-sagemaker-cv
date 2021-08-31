GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

python -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node ${GPU_COUNT} \
    train_yolo.py 
