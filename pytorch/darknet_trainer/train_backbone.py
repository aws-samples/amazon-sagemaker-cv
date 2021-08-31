import os
import json
from ast import literal_eval
import argparse
import torch
from apex import amp
from data.imagenet_loader import load_data
from data.utils import Prefetcher
from model.backbone import darknet53
from engine.scheduler import CosineAnnealingWarmUpRestarts
from engine.trainer import Trainer
from utils.comm import all_gather
from statistics import mean
# from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def parse():
    parser = argparse.ArgumentParser(description='Load model configuration')
    parser.add_argument('--batch_size', help='Global Batch Size', default=2048, type=int)
    parser.add_argument('--learning_rate', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', help='Weight Decay', default=0.0001, type=float)
    parser.add_argument('--opt_level', help='Optimization Level', default='O1', type=str)
    parser.add_argument('--nhwc', help='Channel Last', default='False', type=str)
    parser.add_argument('--epochs', help='Number of Epochs', default=120, type=int)
    parser.add_argument('--mixup_alpha', help='Mix up degree', default=0.2, type=float)
    parser.add_argument('--train_data_dir', help='data directory', default='/opt/ml/input/data/imagenet/train/', type=str)
    parser.add_argument('--val_data_dir', help='data directory', default='/opt/ml/input/data/imagenet/validation/', type=str)
    parser.add_argument('--output_dir', help='output directory', default='/opt/ml/output/', type=str)
    parser.add_argument('--checkpoint_file', help='checkpoint weights to load', default='', type=str)
    parsed, _ = parser.parse_known_args()
    return parsed

def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    main_rank = rank==0
    torch.cuda.set_device(local_rank)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    steps_per_epoch = 1280000//args.batch_size
    iterations = steps_per_epoch * args.epochs
    
    if main_rank:
        print(f"Using Torch For Distributed Training on {world_size} GPUs")
    
    dist.init_process_group(
                backend="nccl", init_method="env://",
            )
    
    assert args.batch_size%world_size==0
    local_batch_size = args.batch_size//world_size
    
    train_iterator = Prefetcher(load_data(args.train_data_dir, local_batch_size), device, NHWC=args.nhwc, 
                                fp16=True if args.opt_level in ['O2', 'O3'] else False)
    val_iterator = Prefetcher(load_data(args.val_data_dir, local_batch_size, train=False), device, NHWC=args.nhwc, 
                              fp16=True if args.opt_level in ['O2', 'O3'] else False)
    
    model = darknet53(1000)
    model = model.to(device)
    
    if args.checkpoint_file!='':
        weights = torch.load(args.checkpoint_file)
        weights = {i.replace('module.',''): j for i,j in weights.items()}
        model.load_state_dict(weights)
    
    if args.nhwc:
        model = model.to(memory_format=torch.channels_last)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * args.batch_size/256, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * args.batch_size/256, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, iterations, T_up=500, eta_max=args.learning_rate * args.batch_size/256)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    
    model = DDP(model, device_ids=[local_rank])
    # model = DDP(model, delay_allreduce=True)
    
    trainer = Trainer(model, loss_fn, optimizer, scheduler, args.mixup_alpha)
    
    if main_rank:
        print(f"Training for {args.epochs} epochs.")
    for epoch in range(args.epochs):
        if main_rank:
            print(f"Starting Epoch {epoch+1}")
            print(f"Training for {steps_per_epoch} steps")
        trainer.train_epoch(train_iterator, log=main_rank, steps=steps_per_epoch)
        if main_rank:
            print(f"Evaluating Epoch {epoch+1}")
        accuracy = mean(all_gather(trainer.eval_epoch(val_iterator)))
        if main_rank:
            print(f"Epoch {epoch+1} eval accuracy: {accuracy*100}")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "epoch_{}.ph".format(epoch+1)))

if __name__=="__main__":
    args = parse()
    main(args)