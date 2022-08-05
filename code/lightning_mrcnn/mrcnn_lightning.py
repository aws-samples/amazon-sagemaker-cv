import os
import gc
import functools

import torch
import torch.distributed as dist
import pytorch_lightning as pl

from optimizer import make_novograd_optimizer
from scheduler import make_cosine_lr_scheduler
from config.defaults import _C as cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.coco import HybridDataLoader3
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import (synchronize, 
                                           get_rank, 
                                           is_main_process, 
                                           get_world_size, 
                                           is_main_evaluation_process)
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.engine.trainer import Prefetcher
from fp16_optimizer import FP16_Optimizer
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier
import apex_C, amp_C
import apex
from apex.parallel import DistributedDataParallel as DDP
from cuda_graph import build_graph
from maskrcnn_benchmark.engine.tester import test
from coco_eval import mlperf_test_early_exit

class MaskRCNN(pl.LightningModule):
    
    def __init__(self, cfg, seed=None, random_number_generator=None, *args, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.__dict__.update(locals())
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        # self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.distributed = self.world_size > 1
        if self.distributed:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0
        self.model = build_detection_model(self.cfg)
        self.images_per_gpu_train = self.cfg.SOLVER.IMS_PER_BATCH // self.world_size
        self.images_per_gpu_test = self.cfg.TEST.IMS_PER_BATCH // self.world_size
        _ = self.model.to(torch.device(self.cfg.MODEL.DEVICE))
        self.is_fp16 = (cfg.DTYPE == "float16")
        if self.is_fp16:
            # convert model to FP16
            self.model.half()
        if self.cfg.USE_CUDA_GRAPH:
            self.model, self.shapes = build_graph(self.model, 
                                                  self.cfg, 
                                                  self.images_per_gpu_train, 
                                                  self.images_per_gpu_test, 
                                                  torch.device(self.cfg.MODEL.DEVICE))
        else:
            min_size = self.cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MIN_SIZE_TRAIN, tuple) else self.cfg.INPUT.MIN_SIZE_TRAIN
            max_size = self.cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MAX_SIZE_TRAIN, tuple) else self.cfg.INPUT.MAX_SIZE_TRAIN
            divisibility = max(1, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
            shapes_per_orientation = self.cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION
            min_size = ((min_size + divisibility - 1) // divisibility) * divisibility
            max_size = ((max_size + divisibility - 1) // divisibility) * divisibility
            size_range = (max_size - min_size) // divisibility
            self.shapes = []
            for i in range(0,shapes_per_orientation):
                size = min_size + ((i+1) * size_range // shapes_per_orientation) * divisibility
                self.shapes.append( (min_size, size) )
                self.shapes.append( (size, min_size) )
        
        if self.cfg.SOLVER.OPTIMIZER=="NovoGrad":
            self.optimizer = make_novograd_optimizer(self.cfg, self.model)
        else:
            self.optimizer = make_optimizer(self.cfg, self.model)
        if self.cfg.SOLVER.LR_SCHEDULE=="COSINE":
            self.scheduler = make_cosine_lr_scheduler(self.cfg, self.optimizer)
        else:
            self.scheduler = make_lr_scheduler(self.cfg, self.optimizer)
        gc.disable()
        if self.distributed:
            # master rank broadcasts parameters
            params = list(self.model.parameters())
            flat_params = apex_C.flatten(params)
            torch.distributed.broadcast(flat_params, 0)
            overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
            multi_tensor_applier(
                    amp_C.multi_tensor_scale,
                    overflow_buf,
                    [apex_C.unflatten(flat_params, params), params],
                    1.0)
            #self.model = DDP(self.model, delay_allreduce=True)
        self.arguments = {}
        self.arguments["iteration"] = 0
        self.arguments["nhwc"] = self.cfg.NHWC
        self.arguments['ims_per_batch'] = self.cfg.SOLVER.IMS_PER_BATCH
        self.arguments["distributed"] = self.distributed
        self.arguments["max_annotations_per_image"] = self.cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE
        self.arguments["dedicated_evaluation_ranks"] = 0
        self.arguments["num_training_ranks"] = self.world_size
        self.arguments["training_comm"] = None
        self.arguments["images_per_gpu_train"] = self.images_per_gpu_train
        self.arguments["use_synthetic_input"] = self.cfg.DATALOADER.USE_SYNTHETIC_INPUT
        self.arguments["enable_nsys_profiling"] = self.cfg.ENABLE_NSYS_PROFILING
        self.output_dir = self.cfg.OUTPUT_DIR
        self.save_to_disk = self.rank == 0
        self.checkpointer = DetectronCheckpointer(
                self.cfg, self.model, self.optimizer, self.scheduler, self.output_dir, self.save_to_disk
            )
        self.arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS
        self.extra_checkpoint_data = self.checkpointer.load(self.cfg.MODEL.WEIGHT, self.cfg.NHWC)
        self.arguments.update(self.extra_checkpoint_data)
        if cfg.MODEL.BACKBONE.DONT_RECOMPUTE_SCALE_AND_BIAS:
            self.model.compute_scale_bias()
        if self.is_fp16:
            if self.cfg.SOLVER.OPTIMIZER=="NovoGrad":
                self.optimizer = apex.fp16_utils.fp16_optimizer.FP16_Optimizer(self.optimizer, 
                                                                            dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(self.optimizer, 
                                           dynamic_loss_scale=True, 
                                           dynamic_loss_scale_window=self.cfg.DYNAMIC_LOSS_SCALE_WINDOW)
        if self.distributed:
            self.params = [p for p in self.model.parameters() if p.requires_grad]
            self.overflow_buf = torch.zeros([1], dtype=torch.int32, device="cuda")
            self.gradient_scaler = 1.0 / self.world_size
        
        # Setup MLPerf evaluation
        self.iters_per_epoch = 118287//self.cfg.SOLVER.IMS_PER_BATCH #17500 
        self._iter = 0
        self.per_iter_callback_fn = functools.partial(mlperf_test_early_exit,
                                                iters_per_epoch=self.iters_per_epoch,
                                                tester=functools.partial(test, cfg=self.cfg, shapes=self.shapes),
                                                model=self.model,
                                                distributed=self.distributed,
                                                min_bbox_map=self.cfg.MLPERF.MIN_BBOX_MAP,
                                                min_segm_map=self.cfg.MLPERF.MIN_SEGM_MAP,
                                                world_size=self.world_size)
            
    def configure_optimizers(self):
        #self.optimizer.zero_grad()
        return self.optimizer
    
    def train_dataloader(self):
        hybrid_dataloader = HybridDataLoader3(self.cfg, 
                                              self.images_per_gpu_train, 
                                              self.cfg.DATALOADER.SIZE_DIVISIBILITY, 
                                              self.shapes) if self.cfg.DATALOADER.HYBRID else None
        data_loader, iters_per_epoch = make_data_loader(self.cfg,
                                                        is_train=True,
                                                        is_distributed=self.distributed,
                                                        start_iter=self.arguments["iteration"],
                                                        shapes=self.shapes,
                                                        random_number_generator=self.random_number_generator,
                                                        seed=self.seed,
                                                        hybrid_dataloader=hybrid_dataloader,
                                                        )
        prefetcher = Prefetcher(data_loader, torch.device(self.cfg.MODEL.DEVICE), self.cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE)
        self.prefetcher = prefetcher
        return self.prefetcher
    
    def forward(self, inputs, targets=None):
        return self.model(inputs, targets)
    
    def reduce_loss_dict(self, loss_dict):
        """
        Reduce the loss dictionary from all processes so that process with rank
        0 has the averaged results. Returns a dict with the same fields as
        loss_dict, after reduction.
        """
        world_size = get_world_size()
        if world_size < 2:
            return loss_dict
        with torch.no_grad():
            loss_names = []
            all_losses = []
            for k in sorted(loss_dict.keys()):
                loss_names.append(k)
                all_losses.append(loss_dict[k])
            all_losses = torch.stack(all_losses, dim=0)
            dist.reduce(all_losses, dst=0)
            if dist.get_rank() == 0:
                # only main process gets accumulated, so only divide by
                # world_size in this case
                all_losses /= world_size
            reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
        return reduced_losses
    
    def training_step(self, batch, batch_idx):
        '''if self.distributed and self._iter%self.iters_per_epoch==0:
            self.optimizer.zero_grad()
            # master rank broadcasts parameters
            params = list(self.model.parameters())
            flat_params = apex_C.flatten(params)
            torch.distributed.broadcast(flat_params, 0)
            overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
            multi_tensor_applier(
                    amp_C.multi_tensor_scale,
                    overflow_buf,
                    [apex_C.unflatten(flat_params, params), params],
                    1.0)'''
        self.per_iter_callback_fn(self._iter)
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.optimizer.backward(losses)
        if self.distributed:
            # gradient reduction
            # grads = [p.grad for p in self.model.parameters() if p.requires_grad]
            grads = [p.grad for p in self.params]
            flat_grads = apex_C.flatten(grads)
            # print("Rank {} gradient mean before reduce {}".format(self.rank, float(torch.max(flat_grads))))
            grad_redux = torch.distributed.all_reduce(
                flat_grads, group=None, async_op=True
            )
        self.prefetcher.prefetch_CPU()
        # overflow_buf = torch.zeros([1], dtype=torch.int32, device="cuda")
        if self.distributed:
            grad_redux.wait()
            self.overflow_buf.zero_()
            multi_tensor_applier(
                amp_C.multi_tensor_scale,
                self.overflow_buf,
                [apex_C.unflatten(flat_grads, grads), grads],
                self.gradient_scaler,
            )
            # print("Rank {} gradient mean after reduce {}".format(self.rank, float(torch.max(flat_grads))))
        if self.cfg.SOLVER.GRADIENT_CLIPPING > 0.0:
            torch.nn.utils.clip_grad_norm_(grads, self.cfg.SOLVER.GRADIENT_CLIPPING)
        if self.cfg.SOLVER.OPTIMIZER=="NovoGrad":
            self.optimizer.step()
        else:
            self.optimizer.step(self.overflow_buf)
        self.prefetcher.prefetch_GPU()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        for i,j in loss_dict.items():
            self.log(i, j, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        self._iter += 1
        return loss_dict
        