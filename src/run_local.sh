NUM_GPUS=8

torchrun \
    --nproc_per_node=$NUM_GPUS \
    train_lightning.py