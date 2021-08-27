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

import gpustat
import psutil
import threading
from time import sleep
from collections import defaultdict
from sagemakercv.utils.runner.hooks import Hook
from sagemakercv.utils.dist_utils import master_only
from statistics import mean
import tensorflow as tf
from ..builder import HOOKS

class SystemMonitor(Hook):

    def __init__(self, collection_interval=5, record_interval=50, rolling_mean_interval=12):
        self.running = False
        self.collection_interval = collection_interval
        self.record_interval = record_interval
        self.rolling_mean_interval = rolling_mean_interval
        self.system_metrics = defaultdict(list)

    def gpu_stats(self):
        return {'gpu_{}_{}'.format(i['index'], j): i[k]
                for i in gpustat.GPUStatCollection.new_query().jsonify()['gpus']
                for j, k in zip(['temp', 'util', 'mem'],
                                ['temperature.gpu', 'utilization.gpu', 'memory.used'])}

    def get_system_metrics(self):
        system_stats = dict()
        system_stats['cpu_percent'] = psutil.cpu_percent()
        system_stats.update(self.gpu_stats())
        system_stats['disk_util'] = psutil.disk_usage('/').percent
        system_stats['mem_util'] = psutil.virtual_memory().percent
        return system_stats

    def monitor_system_metrics(self):
        while self.running:
            current_metrics = self.get_system_metrics()
            for i, j in current_metrics.items():
                self.system_metrics[i].append(j)
            sleep(self.collection_interval)

    def start_monitoring(self):
        self.running = True
        self.thread = threading.Thread(target=self.monitor_system_metrics)
        self.thread.start()

    def stop_monitoring(self):
        self.running = False
        self.thread.join()

    @master_only
    def before_run(self, runner):
        self.start_monitoring()

    @master_only
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.record_interval):
            with runner.writer.as_default():
                for var, metric in self.system_metrics.items():
                    tag = '{}/{}'.format('system', var)
                    record = mean(metric[-self.rolling_mean_interval:])
                    tf.summary.scalar(tag, record, step=runner.iter)
        runner.writer.flush()

    @master_only
    def after_run(self, runner):
        self.stop_monitoring()

@HOOKS.register("SystemMonitor")
def build_system_monitor(cfg):
    return SystemMonitor()