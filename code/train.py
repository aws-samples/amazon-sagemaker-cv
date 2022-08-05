import os
import sys
import random
import gc
import random
from time import time
import argparse
sys.path.append('/workspace/maskrcnn/')
import numpy as np
import torch
import pytorch_lightning as pl
from config.defaults import _C as cfg
from lightning_mrcnn.mrcnn_lightning import MaskRCNN
from lightning_mrcnn.strategy import MLPerfStrategy
try:
    import smdistributed.dataparallel.torch.torch_smddp
except:
    pass

from callbacks import PlSageMakerLogger, ProfilerCallback
from maskrcnn_benchmark.utils.comm import (synchronize, 
                                           get_rank, 
                                           is_main_process, 
                                           get_world_size, 
                                           is_main_evaluation_process)
from maskrcnn_benchmark.utils.mlperf_logger import (log_end, 
                                                    log_start, 
                                                    log_event, 
                                                    generate_seeds, 
                                                    broadcast_seeds, 
                                                    barrier, 
                                                    configure_logger)
from maskrcnn_benchmark.utils.async_evaluator import init

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)

num_gpus = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
distributed = num_gpus > 1

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--config-file', default='e2e_mask_rcnn_R_50_FPN_1x.yaml',
                         help="""Configuration file name.""")
    cmdline.add_argument('--dist', default='nccl',
                         help="""distribution type""")
    return cmdline

def unarchive_data(data_dir, target='coco.tar'):
    print("Unarchiving COCO data")
    os.system('tar -xf {0} -C {1}'.format(os.path.join(data_dir, target), data_dir))
    print(os.listdir(data_dir))

def main(config_file, dist):
    init()
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
                backend=dist, init_method="env://"
            )
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
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
    
    if dist=='smddp' and local_rank==0:
        unarchive_data("/opt/ml/input/data/all_data/")

    cfg.merge_from_file(config_file)
    random_number_generator = random.Random(master_seed)
    worker_seeds = generate_seeds(random_number_generator, world_size)
    if world_size > 1:
        worker_seeds = broadcast_seeds(worker_seeds, device='cuda')
    torch.manual_seed(worker_seeds[rank])
    random.seed(worker_seeds[rank])
    np.random.seed(worker_seeds[rank])

    model = MaskRCNN(cfg, seed=master_seed, random_number_generator=random_number_generator)

    trainer_params = {'strategy': MLPerfStrategy(cfg),
                      'progress_bar_refresh_rate': 0,
                      'callbacks': [PlSageMakerLogger(frequency=100)]}

    
    trainer = pl.Trainer(**trainer_params)
    
    start_time = time()
    trainer.fit(model)
    if rank==0:
        print("Training Finished - Training time: {}".format(time() - start_time))

if __name__=='__main__':
    cmdline = parse_args()
    ARGS, unknown_args = cmdline.parse_known_args()
    config_file = os.path.join('./config', ARGS.config_file)
    main(config_file, ARGS.dist)