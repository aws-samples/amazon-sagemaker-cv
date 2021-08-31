# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from .hook import Hook
import os
from sagemakercv.utils.checkpoint import Checkpointer, DetectronCheckpointer
from ..builder import HOOKS

class CheckpointHook(Hook):

    def __init__(self,
                 interval=1,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs
        
    def before_run(self, runner):
        self.checkpointer = Checkpointer(
            runner.model, 
            runner.optimizer, 
            runner.scheduler, 
            runner.work_dir, 
            runner.rank==0
        )
        self.extra_checkpoint_data = self.checkpointer.load(runner.cfg.MODEL.WEIGHT, 
                                                            runner.cfg.OPT_LEVEL=="O4")

    def after_train_epoch(self, runner):
        # Disable for now, causing error on training job
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        if runner.rank==0:
            checkpoint_dir = os.path.join(self.out_dir, "{:03d}".format(runner.epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpointer.save("model_{:07d}".format(runner.iter), nhwc=runner.cfg.OPT_LEVEL=="O4")
        
class DetectronCheckpointHook(CheckpointHook):
        
    # TODO fix optimizer state dict to enable saving
    def before_run(self, runner):
        self.checkpointer = DetectronCheckpointer(
            runner.cfg,
            runner.model,
            None, # runner.optimizer, 
            None, # runner.scheduler, 
            runner.work_dir,
            runner.rank==0
        )
        self.extra_checkpoint_data = self.checkpointer.load(runner.cfg.MODEL.WEIGHT,
                                                            runner.cfg.OPT_LEVEL=="O4")

@HOOKS.register("DetectronCheckpointHook")
def build_detectron_checkpoint_hook(cfg):
    return DetectronCheckpointHook(interval=cfg.SAVE_INTERVAL)
