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
from sagemakercv.utils import Registry

LOSSES = Registry()
OPTIMIZERS = Registry()
SCHEDULERS = Registry()
SAMPLERS = Registry()
TRAINERS = Registry()

# TODO Add losses to builders

def build_scheduler(cfg):
    scheduler = SCHEDULERS[cfg.SOLVER.SCHEDULE](cfg)
    if cfg.SOLVER.WARMUP:
        scheduler = SCHEDULERS[cfg.SOLVER.WARMUP](cfg, scheduler)
    return scheduler

def build_optimizer(cfg):
    scheduler = build_scheduler(cfg)
    optimizer = OPTIMIZERS[cfg.SOLVER.OPTIMIZER](cfg, scheduler)
    if cfg.SOLVER.FP16:
        if int(tf.__version__.split('.')[1])<4:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
        else:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, 
                                                                    dynamic=True, 
                                                                    initial_scale=2 ** 15, 
                                                                    dynamic_growth_steps=2000
                                                                   )
    return optimizer

def build_trainer(cfg, model, optimizer, dist=None):
    return TRAINERS[cfg.SOLVER.TRAINER](cfg, model, optimizer, dist)