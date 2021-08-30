# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .hook import Hook
import os
import apex
from ..builder import HOOKS

class AMP_Hook(Hook):
    
    def __init__(self, opt_level="O1"):
        self.opt_level=opt_level
    
    def before_run(self, runner):
        runner.model, runner.optimizer = apex.amp.initialize(runner.model, runner.optimizer, opt_level=self.opt_level)

@HOOKS.register("AMP_Hook")
def build_amp_hook(cfg):
    return AMP_Hook(opt_level=cfg.SOLVER.OPT_LEVEL)
