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

from sagemakercv.utils.runner.hooks import Hook
from sagemakercv.utils.dist_utils import master_only
import tensorflow as tf
from ..builder import HOOKS

class Profiler(Hook):
    
    def __init__(self, start_iter=1024, stop_iter=1152):
        self.start_iter = start_iter
        self.stop_iter = stop_iter
    
    @master_only
    def after_train_iter(self, runner):
        if runner.iter == self.start_iter:
            tf.profiler.experimental.start(runner.tensorboard_dir)
        elif runner.iter == self.stop_iter:
            tf.profiler.experimental.stop()

@HOOKS.register("Profiler")
def build_profiler(cfg):
    return Profiler()