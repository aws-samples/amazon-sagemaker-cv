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

def build_scheduler(cfg, keras=False):
    scheduler = SCHEDULERS[cfg.SOLVER.SCHEDULE](cfg)

    # todo: add keras compatible warmup scheduler
    if cfg.SOLVER.WARMUP and not keras:
        scheduler = SCHEDULERS[cfg.SOLVER.WARMUP](cfg, scheduler)
    return scheduler

def build_optimizer(cfg, keras=False):
    scheduler = build_scheduler(cfg, keras)
    optimizer = OPTIMIZERS[cfg.SOLVER.OPTIMIZER](cfg, scheduler)

    # keras does loss_scale automatically
    if cfg.SOLVER.FP16 and not keras:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer,
                                                                dynamic=True,
                                                                initial_scale=2 ** 15,
                                                                dynamic_growth_steps=2000
                                                               )
        # optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    return optimizer

def build_trainer(cfg, model, optimizer, dist=None):
    return TRAINERS[cfg.SOLVER.TRAINER](cfg, model, optimizer, dist)