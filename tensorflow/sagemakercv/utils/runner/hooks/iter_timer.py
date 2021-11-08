#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, Open-MMLab. All rights reserved.
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

import time
import datetime
from ..builder import HOOKS

from .hook import Hook


class IterTimerHook(Hook):
    
    def __init__(self, interval=25):
        self.interval = interval
        
    def before_run(self, runner):
        self.start_time = datetime.datetime.now()
        if runner.rank == 0:
            runner.logger.info("Start time: {}".format(str(self.start_time)))
    
    def before_epoch(self, runner):
        self.t = time.time()
    
    def after_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval) and runner.rank == 0:
            iter_time = (time.time() - self.t)/self.interval
            runner.log_buffer.update({'time': iter_time})
            runner.log_buffer.update({'images/s': runner.train_batch_size/iter_time})
            self.t = time.time()
            
    def after_run(self, runner):
        end_time = datetime.datetime.now()
        if runner.rank == 0:
            runner.logger.info("End time: {}".format(str(self.start_time)))
            runner.logger.info("Elapsed time: {}".format(str(end_time-self.start_time)))

@HOOKS.register("IterTimerHook")
def build_iter_timer_hook(cfg):
    return IterTimerHook(interval=cfg.LOG_INTERVAL)