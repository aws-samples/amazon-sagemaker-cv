# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

import pytorch_lightning as pl

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.data import make_data_loader

from fp16_optimizer import FP16_Optimizer

class GeneralizedRCNN(pl.LightningDataModule):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.precompute_rpn_constant_tensors = cfg.PRECOMPUTE_RPN_CONSTANT_TENSORS
        self.graphable = Graphable(cfg)
        self.combined_rpn_roi = Combined_RPN_ROI(cfg)
        self.nhwc = cfg.NHWC
        self.dali = cfg.DATALOADER.DALI
        self.hybrid_loader = cfg.DATALOADER.HYBRID
        self.scale_bias_callables = None
        
        self.cfg = cfg
        self.is_fp16 = (cfg.DTYPE == "float16")
        self.model = build_detection_model(cfg)
        if self.is_fp16:
            self.model.half()

    def compute_scale_bias(self):
        if self.scale_bias_callables is None:
            self.scale_bias_callables = []
            for module in self.graphable.modules():
                if getattr(module, "get_scale_bias_callable", None):
                    #print(module)
                    c = module.get_scale_bias_callable()
                    self.scale_bias_callables.append(c)
        for c in self.scale_bias_callables:
            c()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if not self.hybrid_loader:
            images = to_image_list(images)
            if self.nhwc and not self.dali:
                # data-loader outputs nchw images
                images.tensors = nchw_to_nhwc_transform(images.tensors)
            elif self.dali and not self.nhwc:
                # dali pipeline outputs nhwc images
                images.tensors = nhwc_to_nchw_transform(images.tensors)
        flat_res = self.graphable(images.tensors, images.image_sizes_tensor)
        features, objectness, rpn_box_regression, anchor_boxes, anchor_visibility = flat_res[0:5], list(flat_res[5:10]), list(flat_res[10:15]), flat_res[15], flat_res[16]
        return self.combined_rpn_roi(images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets, features)
    
    def prepare_data(self):
    
    def train_dataloader(self, distributed, random_number_generator, seed): # distributed is passed into train()
        
        if self.cfg.DATALOADER.ALWAYS_PAD_TO_MAX or self.cfg.USE_CUDA_GRAPH:
            min_size = self.cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MIN_SIZE_TRAIN, tuple) else cfg.INPUT.MIN_SIZE_TRAIN
            max_size = self.cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(self.cfg.INPUT.MAX_SIZE_TRAIN, tuple) else cfg.INPUT.MAX_SIZE_TRAIN
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
            
        data_loader, iters_per_epoch = make_data_loader(
            self.cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=0,# does arguments["iteration"] get updated ?
            random_number_generator=random_number_generator,
            seed=seed,
            shapes=shapes,
            hybrid_dataloader=hybrid_dataloader,
        )
        return data_loader
        
    def val_dataloader(self):
    
    def test_dataloader(self):
        
    
    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self.model)
        scheduler = make_lr_scheduler(self.cfg, optimizer)
        
        if self.is_fp16:
            optimizer FP16_Optimizer(optimizer, dynamic_loss_scale=True, dynamic_loss_scale_window=cfg.DYNAMIC_LOSS_SCALE_WINDOW)
        return optimizer, scheduler
    
    def training_step(self):
        return loss
    def validation_step(self):
        return loss