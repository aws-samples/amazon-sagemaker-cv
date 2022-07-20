# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .hook import Hook
import os
import apex
from ..builder import HOOKS
from sagemakercv.training.optimizers import MLPerfFP16Optimizer

class AMP_Hook(Hook):
    
    def __init__(self, opt_level="O1"):
        self.opt_level=opt_level
    
    def before_run(self, runner):
        if self.opt_level=='O4':
            runner.model.half()
            if "MLPerf" in str(runner.optimizer):
                runner.optimizer = MLPerfFP16Optimizer(runner.optimizer, 
                                                   dynamic_loss_scale=True)
            else:
                runner.optimizer = apex.fp16_utils.fp16_optimizer.FP16_Optimizer(runner.optimizer, 
                                                                             dynamic_loss_scale=True)
        elif self.opt_level in ['O0', 'O1', 'O2', 'O3']:
            runner.model, runner.optimizer = apex.amp.initialize(runner.model, runner.optimizer, opt_level=self.opt_level)
        else:
            raise NotImplementedError("Opt level must be one of O0, O1, O2, O3, O4")

@HOOKS.register("AMP_Hook")
def build_amp_hook(cfg):
    return AMP_Hook(opt_level=cfg.OPT_LEVEL)
