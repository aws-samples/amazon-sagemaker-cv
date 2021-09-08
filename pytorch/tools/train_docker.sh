CONFIG=$1
GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

rm -rf /root/model_outputs

python -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node ${GPU_COUNT} \
    tools/train.py \
    --config ${CONFIG}
