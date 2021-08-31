import os
import json
import argparse
import torch
import logging
import sys
import gc
from ast import literal_eval
from sagemakercv.detection.detector import build_detection_model
from utils import (unarchive_data, 
                   get_training_world, 
                   is_sm, 
                   is_sm_dist, 
                   get_herring_world,
                   config_check)
from sagemakercv.training import make_optimizer, make_lr_scheduler
from sagemakercv.data import make_data_loader
from sagemakercv.data.utils import prefetcher
from sagemakercv.utils.checkpoint import DetectronCheckpointer
from sagemakercv.utils.runner import build_hooks, Runner
from sagemakercv.utils.runner.hooks.checkpoint import DetectronCheckpointHook
from sagemakercv.training.trainers import train_step
import apex
from configs import cfg
from tqdm.auto import tqdm
from statistics import mean

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# os.environ['SM_FRAMEWORK_PARAMS'] = "{\"sagemaker_distributed_dataparallel_custom_mpi_options\":\"\",\"sagemaker_distributed_dataparallel_enabled\":true,\"sagemaker_instance_type\":\"ml.p3dn.24xlarge\"}"

# torch.multiprocessing.set_sharing_strategy('file_system')

if (torch._C, '_jit_set_profiling_executor') :
    torch._C._jit_set_profiling_executor(False)
if (torch._C, '_jit_set_profiling_mode') :
    torch._C._jit_set_profiling_mode(False)
    
def parse():
    parser = argparse.ArgumentParser(description='Load model configuration')
    parser.add_argument('--config', help='Configuration file to apply on top of base')
    parsed, _ = parser.parse_known_args()
    return parsed
    
def main(cfg):
    use_smd = cfg.DISTRIBUTION.lower() in ['herring', 'sagemaker', 'smd']
    if int(os.environ.get("RANK", 0))==0:
        logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s', 
                level=logging.DEBUG,
                stream=sys.stdout)
        logger = logging.getLogger("main_process_logger")
    else:
        logger = None
    gc.disable()
    if use_smd:
        print("Using Herring For Distributed Training")
        import smdistributed.dataparallel.torch.distributed as dist
        from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
        torch.cuda.set_device(dist.get_local_rank())
        if dist.get_local_rank()==0:
            print("initializing dist")
        dist.init_process_group()
        if dist.get_local_rank()==0:
            print("dist initialized")
        if dist.get_local_rank()==0:
            sm_args = os.environ.get("SM_HPS")
            if sm_args:
                sm_args = json.loads(sm_args)
                if "unarchive" in sm_args:
                    data_dir = sm_args.pop("unarchive")
                    print("Starting unarchive")
                    unarchive_data(data_dir)
        if dist.get_local_rank()==0:
            print("before barrier")
        dist.barrier()
        if dist.get_local_rank()==0:
            print("after barrier")
    else:    
        print("Using Torch For Distributed Training")
        import torch.distributed as dist
        from apex.parallel import DistributedDataParallel as DDP
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(
                backend="nccl", init_method="env://",
                rank=int(os.environ.get("RANK", 0)),
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
            )
    train_coco_loader, num_iterations = make_data_loader(cfg, is_distributed=True)
    device = torch.device(cfg.MODEL.DEVICE)
    train_iterator = prefetcher(iter(train_coco_loader), device)
    model = build_detection_model(cfg)
    model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    hooks = build_hooks(cfg)
    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16:
        model.half()
    if use_smd:
        model = DDP(model, broadcast_buffers=False)
    else:   
        model = DDP(model, delay_allreduce=True)
    runner = Runner(model, train_step, cfg, device, optimizer, scheduler, logger=logger)
    for hook in hooks:
        runner.register_hook(hook, priority='HIGHEST' if isinstance(hook, DetectronCheckpointHook) else 'NORMAL')
    runner.run(train_iterator, num_iterations)

if __name__=='__main__':
    args = parse()
    cfg.merge_from_file(args.config)
    config_check(cfg)
    cfg.freeze()
    main(cfg)