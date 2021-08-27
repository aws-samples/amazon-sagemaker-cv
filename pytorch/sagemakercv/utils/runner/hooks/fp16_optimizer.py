# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .hook import Hook
import os
import apex
from ..builder import HOOKS
from sagemakercv.training.optimizers import MLPerfFP16Optimizer

class FP16_Hook(Hook):
    
    def before_run(self, runner):
        if "MLPerf" in str(runner.optimizer):
            runner.optimizer = MLPerfFP16Optimizer(runner.optimizer, 
                                                   dynamic_loss_scale=True)
        else:
            runner.optimizer = apex.fp16_utils.fp16_optimizer.FP16_Optimizer(runner.optimizer, 
                                                                             dynamic_loss_scale=True)

@HOOKS.register("FP16_Hook")
def build_fp16_hook(cfg):
    return FP16_Hook()
