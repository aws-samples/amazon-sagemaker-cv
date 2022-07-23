import pytorch_lightning as pl

import random
import math
import os

import torch
import apex_C, amp_C

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.solver import make_lr_scheduler, make_optimizer
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.engine.trainer import Prefetcher, reduce_loss_dict
from maskrcnn_benchmark.data.datasets.coco import HybridDataLoader3

from fp16_optimizer import FP16_Optimizer

import pdb


class LightningGeneralizedRCNN(pl.LightningModule):
    def __init__(self, cfg, args):
        super(LightningGeneralizedRCNN, self).__init__()
        self.model = build_detection_model(cfg)
#         device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
#         if cfg.DTYPE == "float16":
#             self.model.half()
        self.cfg = cfg
        self.is_fp16 = (cfg.DTYPE == "float16")
        if self.is_fp16:
            self.model.half()
#             self.half()
            
        self.arguments=args
        
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        
    @property
    def automatic_optimization(self):
        return False
        
    def forward(self, images, targets):
#         print("FORWARD")
        return self.model(images, targets)
    
    def configure_optimizers(self):
#         print("OPTIMIZER")
        
        optimizer = make_optimizer(self.cfg, self.model)
        scheduler = make_lr_scheduler(self.cfg, optimizer)
        self.scheduler = scheduler
        
        if self.is_fp16:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True, dynamic_loss_scale_window=self.cfg.DYNAMIC_LOSS_SCALE_WINDOW)
            self.optimizer = optimizer
#             pdb.set_trace()
            return self.optimizer
#             return [self.optimizer], [scheduler]
            
        self.optimizer=optimizer
        return [optimizer], [scheduler]
    
    
    def train_dataloader(self):
#         print("TRAINDATALOADER")
        
        if self.cfg.DATALOADER.ALWAYS_PAD_TO_MAX or cfg.USE_CUDA_GRAPH:
            min_size = self.cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MIN_SIZE_TRAIN, tuple) else self.cfg.INPUT.MIN_SIZE_TRAIN
            max_size = self.cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MAX_SIZE_TRAIN, tuple) else self.cfg.INPUT.MAX_SIZE_TRAIN
            divisibility = max(1, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
            shapes_per_orientation = self.cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION

            min_size = ((min_size + divisibility - 1) // divisibility) * divisibility
            max_size = ((max_size + divisibility - 1) // divisibility) * divisibility
            size_range = (max_size - min_size) // divisibility

            shapes = []
            for i in range(0,shapes_per_orientation):
                size = min_size + ((i+1) * size_range // shapes_per_orientation) * divisibility
                shapes.append( (min_size, size) )
                shapes.append( (size, min_size) )
            print(shapes)
        else:
            shapes = None
            
#         hybrid_dataloader = HybridDataLoader3(self.cfg, self.cfg.SOLVER.IMS_PER_BATCH, self.cfg.DATALOADER.SIZE_DIVISIBILITY, shapes) if self.cfg.DATALOADER.HYBRID else None
        hybrid_dataloader = None
            
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        random_number_generator = random.Random(master_seed)
#         pdb.set_trace()

        data_loader, iters_per_epoch = make_data_loader(
                    self.cfg,
                    is_train=True,
                    is_distributed=self.distributed,
                    start_iter=self.arguments["iteration"],
                    random_number_generator=random_number_generator,
                    seed=master_seed,
                    shapes=shapes,
                    hybrid_dataloader=hybrid_dataloader,
                )
#         pdb.set_trace()
#         device = torch.device(self.cfg.MODEL.DEVICE)
        prefetcher = Prefetcher(data_loader, self.device, ) if not self.arguments["use_synthetic_input"] else SyntheticDataLoader(self.device, bs=self.arguments["images_per_gpu_train"], img_h=800, img_w = 1344, annotations_per_image = 10, max_iter = 65535)
    
        self.prefetcher=prefetcher
#         pdb.set_trace()
        return prefetcher
        
        
#     def val_dataloader(self):
#         if self.cfg.DATALOADER.ALWAYS_PAD_TO_MAX or self.cfg.USE_CUDA_GRAPH:
#             min_size = self.cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MIN_SIZE_TRAIN, tuple) else self.cfg.INPUT.MIN_SIZE_TRAIN
#             max_size = self.cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MAX_SIZE_TRAIN, tuple) else self.cfg.INPUT.MAX_SIZE_TRAIN
#             divisibility = max(1, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
#             shapes_per_orientation = self.cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION

#             min_size = ((min_size + divisibility - 1) // divisibility) * divisibility
#             max_size = ((max_size + divisibility - 1) // divisibility) * divisibility
#             size_range = (max_size - min_size) // divisibility

#             shapes = []
#             for i in range(0,shapes_per_orientation):
#                 size = min_size + ((i+1) * size_range // shapes_per_orientation) * divisibility
#                 shapes.append( (min_size, size) )
#                 shapes.append( (size, min_size) )
#             print(shapes)
#         else:
#             shapes = None
            
#         hybrid_dataloader = HybridDataLoader3(self.cfg, self.arguments["images_per_gpu_train"], self.cfg.DATALOADER.SIZE_DIVISIBILITY, shapes) if self.cfg.DATALOADER.HYBRID else None
            
#         master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
#         random_number_generator = random.Random(master_seed)
#         data_loader = make_data_loader(
#                     self.cfg,
#                     is_train=False,
#                     is_distributed=self.distributed,
#                     start_iter=0,
#                     random_number_generator=random_number_generator,
#                     seed=master_seed,
#                     shapes=shapes,
#                     hybrid_dataloader=hybrid_dataloader,
#                 )[0] #???? how to deal
# #         device = torch.device(self.cfg.MODEL.DEVICE)
#         prefetcher = Prefetcher(data_loader, self.device, ) if not self.arguments["use_synthetic_input"] else SyntheticDataLoader(self.device, bs=self.arguments["images_per_gpu_train"], img_h=800, img_w = 1344, annotations_per_image = 10, max_iter = 65535)
        
#         return prefetcher
        
    def training_step(self, batch, batch_idx):
#         print("TRAININGSTEP")
#         optimizer = self.optimizer
        meters = MetricLogger(delimiter="  ")
        
        self.optimizer.zero_grad()
        
        images_per_gpu_train = self.arguments["images_per_gpu_train"]
        
        if self.distributed:
            params = [p for p in self.model.parameters() if p.requires_grad]
            overflow_buf = torch.zeros([1], dtype=torch.int32, device="cuda")
            gradient_scaler = 1.0 / num_training_ranks
        else:
            overflow_buf = None
            
        vss = []

#         images = batch[0]
#         targets = batch[1]

#         images, (a, b, c, d) = batch
        images, targets = batch
#         print("TARGETS")
#         print(targets[0])
#         (a, b, c, d) = targets
#         print(a)
#         print(b)
#         print(c)
#         print(d)
#         pdb.set_trace()
        targets = [targets[0].bbox.unsqueeze(0), torch.tensor([0.0 if i == 0 else 1.0 for i in targets[0].get_field("labels")]).unsqueeze(0).to('cuda'), targets[0].get_field("labels").unsqueeze(0), targets]
#         print(targets)
        if images_per_gpu_train == 1:
            if self.distributed:
                torch.distributed.barrier(
                    group=training_comm
                )  # Sync all processes before training starts to prevent CPUs from getting too far ahead of GPUs
            else:
                torch.cuda.synchronize()

        loss_dict = self(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        print(losses)

        self.optimizer.backward(losses)

        if self.distributed:
            # gradient reduction
            grads = [p.grad for p in params]
            flat_grads = apex_C.flatten(grads)
            grad_redux = torch.distributed.all_reduce(
                flat_grads, group=training_comm, async_op=True
            )

        # At this point we are waiting for kernels launched by cuda graph to finish, so CPU is idle.
        # Take advantage of this by loading next input batch before calling step
#         self.prefetcher.prefetch_CPU()

        if self.distributed:
            grad_redux.wait()
            overflow_buf.zero_()
            multi_tensor_applier(
                amp_C.multi_tensor_scale,
                overflow_buf,
                [apex_C.unflatten(flat_grads, grads), grads],
                gradient_scaler,
            )

        self.optimizer.step(overflow_buf)  # This will sync
#         self.prefetcher.prefetch_GPU()


        if not self.cfg.DISABLE_LOSS_LOGGING and not self.cfg.DISABLE_REDUCED_LOGGING:
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if math.isfinite(losses_reduced):
                meters.update(loss=losses_reduced, **loss_dict_reduced)
        else:
            # Optimization:
            # Cat all meter updates + finite check if they are all single item tensors
            # This reduces number of D2H transfers to 1.
            ks, vs = zip(*[(k, v.unsqueeze(dim=0)) for (k, v) in loss_dict.items()])
            if disable_loss_logging:
                vs = torch.zeros([len(vs)], dtype=torch.float32)
            else:
                vs = list(vs)
                vs.append(losses.unsqueeze(dim=0))
                vs = torch.cat(vs)
            vss.append(vs)
            if will_report_this_iteration:
                vss = torch.stack(vss).cpu()  # will sync
                for vs in vss:
                    vs = [v.item() for v in list(vs.split(split_size=1))]
                    losses_host = vs.pop(-1)
                    if math.isfinite(losses_host):
                        loss_dict = {k: v for (k, v) in zip(ks, vs)}
                        meters.update(loss=losses_host, **loss_dict)
                vss = []
                
        # set_grads_to_none(model)
        self.optimizer.zero_grad()
        self.scheduler.step()
        return losses
    
    
#     def validation_step(self, batch, batch_idx):
        
#         images_per_gpu_train = self.arguments["images_per_gpu_train"]
        
#         images, (targets, _a, _b, _c)= batch
# #         images = batch[0]
# #         targets = batch[1]
#         device = torch.device(self.cfg.MODEL.DEVICE)
# #         images = images.to(device)
# #         print(imagess.image_sizes_tensor)
# #         targets = targets.to(device)
#         if images_per_gpu_train == 1:
#             if self.distributed:
#                 torch.distributed.barrier(
#                     group=self.training_comm
#                 )  # Sync all processes before training starts to prevent CPUs from getting too far ahead of GPUs
#             else:
# #                 torch.cuda.synchronize()
#                 print("")
#         loss_dict = self(images, targets)
#         print(loss_dict)
#         losses = sum(loss for loss in loss_dict.values())
        
#         return losses