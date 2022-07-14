import pytorch_lightning as pl

import random
import os

import torch
import apex_C, amp_C

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler, make_optimizer
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.engine.trainer import Prefetcher
from fp16_optimizer import FP16_Optimizer

class LightningGeneralizedRCNN(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningGeneralizedRCNN, self).__init__()
        self.model = build_detection_model(cfg)
        if cfg.DTYPE == "float16":
            self.model.half()
        self.cfg = cfg
        self.is_fp16 = (cfg.DTYPE == "float16")
        if self.is_fp16:
            self.model.half()
            
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.distributed = self.num_gpus > 1
        
        world_size = 1 # where is this from??
        
        dedicated_evaluation_ranks = max(0,cfg.DEDICATED_EVALUATION_RANKS)
        num_training_ranks = world_size - dedicated_evaluation_ranks
        images_per_gpu_train = cfg.SOLVER.IMS_PER_BATCH
        
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

            if dedicated_evaluation_ranks > 0:
                # create nccl comm for training ranks
                training_ranks = [i for i in range(num_training_ranks)]
                self.training_comm = torch.distributed.new_group(ranks=training_ranks)
                dummy = torch.ones([1], device='cuda')
                torch.distributed.all_reduce(dummy, group=self.training_comm) # wake up new comm

                # create nccl comm for evaluation ranks
                evaluation_ranks = [i+num_training_ranks for i in range(dedicated_evaluation_ranks)]
                evaluation_comm = torch.distributed.new_group(ranks=evaluation_ranks)
                dummy.fill_(1)
                torch.distributed.all_reduce(dummy, group=evaluation_comm) # wake up new comm
        
        self.arguments = {}
        self.arguments["iteration"] = 0
        self.arguments["nhwc"] = cfg.NHWC
        self.arguments['ims_per_batch'] = cfg.SOLVER.IMS_PER_BATCH
        self.arguments["distributed"] = self.distributed
        self.arguments["max_annotations_per_image"] = self.cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE
        self.arguments["dedicated_evaluation_ranks"] = dedicated_evaluation_ranks
        self.arguments["num_training_ranks"] = num_training_ranks
        self.arguments["training_comm"] = None if dedicated_evaluation_ranks == 0 else training_comm
        self.arguments["images_per_gpu_train"] = images_per_gpu_train
        self.arguments["use_synthetic_input"] = cfg.DATALOADER.USE_SYNTHETIC_INPUT
        assert not (cfg.DATALOADER.USE_SYNTHETIC_INPUT and cfg.DATALOADER.HYBRID), "USE_SYNTHETIC_INPUT and HYBRID can't both be used together"
        self.arguments["enable_nsys_profiling"] = cfg.ENABLE_NSYS_PROFILING
#         output_dir = cfg.OUTPUT_DIR

#         save_to_disk = get_rank() == 0
#         checkpointer = DetectronCheckpointer(
#             cfg, self.model, optimizer, scheduler, output_dir, save_to_disk
#         )
#         arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS

#         extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, cfg.NHWC)
#         arguments.update(extra_checkpoint_data)

    def forward(self, images, targets):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        self.optimizer = make_optimizer(self.cfg, self.model)
        
#         if self.is_fp16:
#             self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True, dynamic_loss_scale_window=self.cfg.DYNAMIC_LOSS_SCALE_WINDOW)
# TypeError: FP16_Optimizer is not an Optimizer
            
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer)
        return {"optimizer": self.optimizer,"lr_scheduler": self.scheduler}
    
    def train_dataloader(self):
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
        
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        distributed = num_gpus > 1
            
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        random_number_generator = random.Random(master_seed)
        data_loader = make_data_loader(
                    self.cfg,
                    is_train=True,
                    is_distributed=distributed,
                    start_iter=0,
                    random_number_generator=random_number_generator,
                    seed=master_seed,
                    shapes=shapes,
                    hybrid_dataloader=None,
                )
        device = torch.device(self.cfg.MODEL.DEVICE)
        prefetcher = Prefetcher(data_loader, device, self.arguments["max_annotations_per_image"]) if not self.arguments["use_synthetic_input"] else SyntheticDataLoader(device, bs=self.arguments["images_per_gpu_train"], img_h=800, img_w = 1344, annotations_per_image = 10, max_iter = 65535)
        
        return prefetcher
        
        
    def val_dataloader(self):
        if self.cfg.DATALOADER.ALWAYS_PAD_TO_MAX or self.cfg.USE_CUDA_GRAPH:
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
            
        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        distributed = num_gpus > 1
            
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
        random_number_generator = random.Random(master_seed)
        data_loader = make_data_loader(
                    self.cfg,
                    is_train=False,
                    is_distributed=self.distributed,
                    start_iter=0,
                    random_number_generator=random_number_generator,
                    seed=master_seed,
                    shapes=shapes,
                    hybrid_dataloader=None,
                )
        device = torch.device(self.cfg.MODEL.DEVICE)
        prefetcher = Prefetcher(data_loader, device, self.arguments["max_annotations_per_image"]) if not self.arguments["use_synthetic_input"] else SyntheticDataLoader(device, bs=self.arguments["images_per_gpu_train"], img_h=800, img_w = 1344, annotations_per_image = 10, max_iter = 65535)
        
        return prefetcher
        
    def training_step(self, batch, batch_idx):

        optimizer = self.optimizer
        optimizer.zero_grad()
        
        if distributed:
            params = [p for p in model.parameters() if p.requires_grad]
            overflow_buf = torch.zeros([1], dtype=torch.int32, device="cuda")
            gradient_scaler = 1.0 / num_training_ranks
        else:
            overflow_buf = None

        images = batch[0]
        targets = batch[1]

        if images_per_gpu_train == 1:
            if distributed:
                torch.distributed.barrier(
                    group=training_comm
                )  # Sync all processes before training starts to prevent CPUs from getting too far ahead of GPUs
            else:
                torch.cuda.synchronize()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.backward(losses)

        if distributed:
            # gradient reduction
            grads = [p.grad for p in params]
            flat_grads = apex_C.flatten(grads)
            grad_redux = torch.distributed.all_reduce(
                flat_grads, group=training_comm, async_op=True
            )

        # At this point we are waiting for kernels launched by cuda graph to finish, so CPU is idle.
        # Take advantage of this by loading next input batch before calling step


        if distributed:
            grad_redux.wait()
            overflow_buf.zero_()
            multi_tensor_applier(
                amp_C.multi_tensor_scale,
                overflow_buf,
                [apex_C.unflatten(flat_grads, grads), grads],
                gradient_scaler,
            )

        optimizer.step(overflow_buf)  # This will sync

        # set_grads_to_none(model)
        optimizer.zero_grad()
        scheduler.step()
        return losses
    
    
    def validation_step(self, batch, batch_idx):
        
        images_per_gpu_train = self.arguments["images_per_gpu_train"]
        
        
        images = batch[0]
        targets = batch[1]
        if images_per_gpu_train == 1:
            if self.distributed:
                torch.distributed.barrier(
                    group=self.training_comm
                )  # Sync all processes before training starts to prevent CPUs from getting too far ahead of GPUs
            else:
                torch.cuda.synchronize()

        loss_dict = self.model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        
        return losses