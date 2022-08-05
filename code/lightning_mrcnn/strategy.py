import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities.distributed import (
    _get_process_group_backend_from_env,
    all_gather_ddp_if_available,
    get_default_process_group_backend_for_device,
    ReduceOp,
    distributed_available,
    sync_ddp_if_available,
)
from pytorch_lightning.utilities.distributed import group as _group

class MLPerfStrategy(Strategy):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._root_device = torch.device(self.cfg.MODEL.DEVICE)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = 0
        if self.world_size>1:
            self.global_rank = torch.distributed.get_rank()
    
    def setup_environment(self):
        pass
    
    def setup(self, trainer):
        pass
    
    def teardown(self):
        pass
    
    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def barrier(self, *args, **kwargs):
        '''if not distributed_available():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()'''
        return
            
    def broadcast(self, obj, src = 0):
        '''if not distributed_available():
            return obj
        obj = [obj]
        if self.global_rank != src:
            obj = [None]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]'''
        return obj
    
    def dispatch(self, trainer):
        pass
    
    def all_gather(self, tensor, group, sync_grads = False):
        #return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)
        return tensor
    
    @property
    def is_global_zero(self):
        return self.global_rank==0
    
    def model_to_device(self):
        self.model.to(self.root_device)
    
    def reduce(self, tensor, group = None, reduce_op = "mean"):
        '''if isinstance(tensor, Tensor):
            tensor = sync_ddp_if_available(tensor, group, reduce_op=reduce_op)'''
        return tensor
    
    @property
    def root_device(self):
        return self._root_device