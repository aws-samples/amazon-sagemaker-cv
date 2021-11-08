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

@SCHEDULERS.register("PiecewiseConstantDecay")
def build_piecewise_scheduler(cfg):
    assert len(cfg.SOLVER.DECAY_STEPS)==len(cfg.SOLVER.DECAY_LR)
    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay
    return scheduler(cfg.SOLVER.DECAY_STEPS,
                     [cfg.SOLVER.LR] + [cfg.SOLVER.LR * decay for decay in cfg.SOLVER.DECAY_LR])

@SCHEDULERS.register("CosineDecay")
def build_cosine_scheduler(cfg):
    scheduler = tf.keras.experimental.CosineDecay
    return scheduler(cfg.SOLVER.LR,
                     cfg.SOLVER.MAX_ITERS,
                     cfg.SOLVER.ALPHA)
