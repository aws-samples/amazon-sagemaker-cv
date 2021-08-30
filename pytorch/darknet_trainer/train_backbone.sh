GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

python -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node $GPU_COUNT \
    train_backbone.py \
    --batch_size 512 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --opt_level O1 \
    --nhwc False \
    --epochs 25 \
    --mixup_alpha 0.0 \
    --train_data_dir /workspace/data/imagenet/train \
    --val_data_dir /workspace/data/imagenet/validation \
    --output_dir /workspace/data/model_output/ \
    --checkpoint_file /workspace/data/model_output/epoch_109.ph