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

import tensorflow as tf
from ...builder import SCHEDULERS

class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear or exponential warmup
    """
    
    def __init__(self, scheduler, warmup_ratio, warmup_steps, 
                 overlap=False, warmup_type='linear', dtype=tf.float32):
        super(LinearWarmup, self).__init__()
        self.scheduler = scheduler
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.warmup_type = warmup_type
        self.dtype = dtype
        self.scheduler_learning_rate = scheduler(0)
        self.initial_learning_rate = tf.cast(self.scheduler_learning_rate * self.warmup_ratio, dtype)
        self.overlap = overlap
        
    def compute_linear_warmup(self, step):
        return ((self.scheduler_learning_rate*step) + (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
    
    @tf.function(experimental_relax_shapes=True)
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp>=self.warmup_steps:
            if self.overlap:
                return self.scheduler(global_step_recomp)
            return self.scheduler(global_step_recomp - self.warmup_steps)
        return self.compute_linear_warmup(global_step_recomp)
    
    def get_config(self):
        scheduler_config = self.scheduler.get_config()
        scheduler_config['initial_learning_rate'] = self.initial_learning_rate
        scheduler_config['warmup_steps'] = self.warmup_steps
        scheduler_config['warmup_type'] = self.warmup_type
        scheduler_config['overlap'] = self.overlap
        return scheduler_config

@SCHEDULERS.register("LinearWarmup")
def build_linear_warmup(cfg, scheduler):
    return LinearWarmup(scheduler=scheduler,
                        warmup_ratio=cfg.SOLVER.WARM_UP_RATIO,
                        warmup_steps=cfg.SOLVER.WARMUP_STEPS,
                        overlap=cfg.SOLVER.WARMUP_OVERLAP)