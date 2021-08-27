#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Amazon Web Services. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tensorflow as tf
from sagemakercv.utils.dist_utils import master_only
from .hook import Hook
from ..builder import HOOKS

class CheckpointHook(Hook):

    def __init__(self,
                 interval=-1,
                 save_optimizer=True,
                 out_dir=None,
                 backbone_checkpoint=None,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.backbone_checkpoint = backbone_checkpoint
        self.args = kwargs
        
    @master_only
    def before_run(self, runner):
        if self.backbone_checkpoint:
            runner.logger.info('Loading checkpoint from %s...', self.backbone_checkpoint)
            chkp = tf.compat.v1.train.NewCheckpointReader(self.backbone_checkpoint)
            weights = [chkp.get_tensor(i) for i in ['/'.join(i.name.split('/')[-2:]).split(':')[0] \
                                                for i in runner.trainer.model.layers[0].weights]]
            runner.trainer.model.layers[0].set_weights(weights)
            return

    @master_only
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        if not self.out_dir:
            self.out_dir = runner.work_dir
        checkpoint_dir = os.path.join(self.out_dir, "{:03d}".format(runner.epoch))
        os.makedirs(checkpoint_dir)
        filepath = os.path.join(checkpoint_dir, 'model.h5')
        runner.trainer.model.save_weights(filepath)
        runner.logger.info('Saved checkpoint at: {}'.format(filepath))
        
    @master_only
    def after_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir
        checkpoint_dir = os.path.join(self.out_dir, 'trained_model')
        os.makedirs(checkpoint_dir)
        filepath = os.path.join(checkpoint_dir, 'model.h5')
        runner.trainer.model.save_weights(filepath)
        runner.logger.info('Saved checkpoint at: {}'.format(filepath))

@HOOKS.register("CheckpointHook")
def build_checkpoint_hook(cfg):
    return CheckpointHook(interval=cfg.SOLVER.CHECKPOINT_INTERVAL,
                          backbone_checkpoint=cfg.PATHS.WEIGHTS)
