import os
import json
import argparse
import sys
import warnings
import random
from pathlib import Path
from ast import literal_eval
warnings.filterwarnings('ignore')

import torch
import torchvision as tv
import pytorch_lightning as pl
# import webdataset as wds
# from sm_resnet.models import ResNet
# from sm_resnet.callbacks import PlSageMakerLogger, ProfilerCallback, SMDebugCallback
# from sm_resnet.utils import get_training_world, get_rank
# import smdebug.pytorch as smd
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig
from smdebug.core.collection import CollectionKeys
# from smdebug.core.utils import check_sm_training_env

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.mlperf_logger import configure_logger
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process

from mrcnn_lightning_strategy import MRCNNLightningStrategy
from rcnn_lightning import LightningGeneralizedRCNN

#rank = os.environ.get("LOCAL_RANK", '0')
#os.environ['CUDA_VISIBLE_DEVICES'] = rank

# world = get_training_world()

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--train_file_dir', default='/opt/ml/input/data/train/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--validation_file_dir', default='/opt/ml/input/data/val/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--max_epochs', default=20, type=int,
                         help="""Number of epochs.""")
    cmdline.add_argument('--num_classes', default=1000, type=int,
                         help="""Number of classes.""")
    cmdline.add_argument('--resnet_version', default=50, type=int,
                         help="""Resnet version.""")
    cmdline.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
                         help="""Base learning rate.""")
    cmdline.add_argument('-b', '--batch_size', default=128, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--warmup_epochs', default=1, type=int,
                         help="""Number of epochs for learning rate warmup""")
    cmdline.add_argument('--mixup_alpha', default=0.0, type=float,
                         help="""Extent of convex combination for training mixup""")
    cmdline.add_argument('--optimizer', default='adamw', type=str,
                         help="""Optimizer type""")
    cmdline.add_argument('--strategy', default='horovod', type=str,
                         help="""Distribution strategy""")
    cmdline.add_argument('--precision', default=16, type=int,
                         help="""Floating point precision""")
    cmdline.add_argument('--profiler_start', default=128, type=int,
                         help="""Profiler start step""")
    cmdline.add_argument('--profiler_steps', default=32, type=int,
                         help="""Profiler steps""")
    cmdline.add_argument('--dataloader_workers', default=4, type=int,
                         help="""Number of data loaders""")
    cmdline.add_argument('--debugging_output', default='/opt/ml/debugger/',
                         help="""Path to dataset in WebDataset format.""")
    cmdline.add_argument('--train_batches', default=1, type=int,
                         help="""Number of batches to use for each training epoch""")
    return cmdline
    
def main():
    
    cfg.merge_from_file("configs/e2e_mask_rcnn_R_50_FPN_1x_1_node_test_local.yaml")
    
    configure_logger(constants.MASKRCNN)
    
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
            "--config-file",
            default="",
            metavar="FILE",
            help="path to config file",
            type=str,
        )
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args_distributed = num_gpus >1
    
    if args_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        # setting seeds - needs to be timed, so after RUN_START
        if is_main_process():
            master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
            seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))
        else:
            seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))

        torch.distributed.broadcast(seed_tensor, 0)
        master_seed = int(seed_tensor.item())
    else:
        world_size = 1
        rank = 0
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        
    distributed = num_gpus > 1
    dedicated_evaluation_ranks = max(0,cfg.DEDICATED_EVALUATION_RANKS)
    num_training_ranks = world_size - dedicated_evaluation_ranks
    num_evaluation_ranks = world_size if dedicated_evaluation_ranks == 0 else dedicated_evaluation_ranks

    images_per_gpu_train = cfg.SOLVER.IMS_PER_BATCH // num_training_ranks
    images_per_gpu_test = cfg.TEST.IMS_PER_BATCH // num_evaluation_ranks
    
    if distributed:
#         # master rank broadcasts parameters
#         params = list(model.parameters())
#         flat_params = apex_C.flatten(params)
#         torch.distributed.broadcast(flat_params, 0)
#         overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
#         multi_tensor_applier(
#                 amp_C.multi_tensor_scale,
#                 overflow_buf,
#                 [apex_C.unflatten(flat_params, params), params],
#                 1.0)

        if dedicated_evaluation_ranks > 0:
            # create nccl comm for training ranks
            training_ranks = [i for i in range(num_training_ranks)]
            training_comm = torch.distributed.new_group(ranks=training_ranks)
            dummy = torch.ones([1], device='cuda')
            torch.distributed.all_reduce(dummy, group=training_comm) # wake up new comm

            # create nccl comm for evaluation ranks
            evaluation_ranks = [i+num_training_ranks for i in range(dedicated_evaluation_ranks)]
            evaluation_comm = torch.distributed.new_group(ranks=evaluation_ranks)
            dummy.fill_(1)
            torch.distributed.all_reduce(dummy, group=evaluation_comm) # wake up new comm

    arguments = {}
    arguments["iteration"] = 0
    arguments["nhwc"] = cfg.NHWC
    arguments['ims_per_batch'] = cfg.SOLVER.IMS_PER_BATCH
    arguments["distributed"] = distributed
    arguments["max_annotations_per_image"] = cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE
    arguments["dedicated_evaluation_ranks"] = dedicated_evaluation_ranks
    arguments["num_training_ranks"] = num_training_ranks
    arguments["training_comm"] = None if dedicated_evaluation_ranks == 0 else training_comm
    arguments["images_per_gpu_train"] = images_per_gpu_train
    arguments["use_synthetic_input"] = cfg.DATALOADER.USE_SYNTHETIC_INPUT
    assert not (cfg.DATALOADER.USE_SYNTHETIC_INPUT and cfg.DATALOADER.HYBRID), "USE_SYNTHETIC_INPUT and HYBRID can't both be used together"
    arguments["enable_nsys_profiling"] = cfg.ENABLE_NSYS_PROFILING
    
    model = LightningGeneralizedRCNN(cfg, arguments)
    model.to('cuda')
    trainer = pl.Trainer(accelerator="gpu", devices=8, strategy = MRCNNLightningStrategy(), enable_checkpointing=False)


    trainer.fit(model)

if __name__=='__main__':
#     cmdline = parse_args()
#     ARGS, unknown_args = cmdline.parse_known_args()
    main()