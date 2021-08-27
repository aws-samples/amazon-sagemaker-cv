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
import re
import tensorflow as tf
from s3fs import S3FileSystem
from concurrent.futures import ThreadPoolExecutor
from sagemakercv.utils.dist_utils import master_only
from .base import LoggerHook
from ...builder import HOOKS

class TensorboardMetricsLogger(LoggerHook):

    def __init__(self,
                 name='metrics',
                 re_match='.*loss',
                 interval=100,
                 image_interval=None,
                 ignore_last=True,
                 reset_flag=True):
        super(TensorboardMetricsLogger, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.name = name
        if isinstance(re_match, str):
            self.re_match = [re_match]
        else:
            self.re_match = re_match
    
    @master_only
    def log(self, runner):
        matched_tensors = []
        for expression in self.re_match:
            matched_tensors.extend(list(filter(re.compile(expression).match, 
                                               runner.log_buffer.output.keys())))
        writer = tf.summary.create_file_writer(runner.tensorboard_dir)
        with writer.as_default():
            for var in matched_tensors:
                tag = '{}/{}'.format(self.name, var)
                record = runner.log_buffer.output[var]
                if isinstance(record, str):
                    tf.summary.text(tag, record, step=runner.iter)
                else:
                    tf.summary.scalar(tag, record, step=runner.iter) 
        writer.close()

@HOOKS.register("TensorboardMetricsLogger")
def build_tensorboard_metrics_logger(cfg):
    return TensorboardMetricsLogger(interval=cfg.LOG_INTERVAL)