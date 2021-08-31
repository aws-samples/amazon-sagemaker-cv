# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch
import apex

from .optimizers.schedulers import WarmupMultiStepLR
from .optimizers.schedulers import CosineAnnealingWarmUpRestarts

from .optimizers import MLPerfFusedSGD
from apex.optimizers import FusedSGD
from apex.optimizers import FusedAdam, FusedLAMB
from .optimizers.fused_novograd import FusedNovoGrad

def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    bias_params = []
    bias_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    bias_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "bias" in key:
            bias_params.append(value)
        else:
            params.append(value)
    
    if cfg.SOLVER.OPTIMIZER == "NovoGrad":
        optimizer = FusedNovoGrad(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2), eps=1e-7, 
            grad_averaging=False, init_zero=False, reg_inside_moment=True, bias_correction=True)
    elif cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = FusedAdam(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2), eps=1e-7, 
            grad_averaging=False, bias_correction=True)
    elif cfg.SOLVER.OPTIMIZER == "Lamb":
        optimizer = FusedLAMB(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2), eps=1e-7, 
            grad_averaging=False, bias_correction=True)
    elif cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = FusedSGD(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER == "MLPerfSGD":
        optimizer = MLPerfFusedSGD(
            [
                {"params": params, "lr": lr, "weight_decay": weight_decay},
                {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
            ],
            lr, momentum=cfg.SOLVER.MOMENTUM)
    else:
        raise NotImplementedError("Available optimizers are SGD, MLPerfSGD, and NovoGrad")

    return optimizer

def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_SCHEDULE == "COSINE":
        return CosineAnnealingWarmUpRestarts(
            optimizer, # Novograd
            T_0 = cfg.SOLVER.MAX_ITER, # total steps solver.max_iter
            eta_max = cfg.SOLVER.BASE_LR, # max lr or base lr init_lr
            alpha = cfg.SOLVER.ALPHA,
            T_up = cfg.SOLVER.WARMUP_ITERS, # warmup steps  , warmupsteps
        )
    elif cfg.SOLVER.LR_SCHEDULE == "MULTISTEP":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

