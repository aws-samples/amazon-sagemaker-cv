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

import datetime
import os.path as osp
from collections import OrderedDict
from sagemakercv.utils.fileio import dump
from .base import LoggerHook
from ...builder import HOOKS

class TextLoggerHook(LoggerHook):

    def __init__(self, interval=25, ignore_last=True, reset_flag=False):
        super(TextLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.time_sec_tot = 0

    def before_run(self, runner):
        super(TextLoggerHook, self).before_run(runner)
        self.start_iter = runner.iter
        self.json_log_path = osp.join(runner.work_dir,
                                      '{}.log.json'.format(runner.timestamp))

    def _log_info(self, log_dict, runner):
        if runner.mode == 'train':
            log_str = 'Epoch [{}][{}/{}]\tlr: {:.5f}, '.format(
                log_dict['epoch'], log_dict['iter'], runner.num_examples,
                log_dict['lr'])
            if 'time' in log_dict.keys():
                eta_sec = log_dict['time'] * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += 'eta: {}, '.format(eta_str)
                log_str += ('step time: {:.3f}, '.format(log_dict['time']))
        else:
            log_str = 'Epoch({}) [{}][{}]\t'.format(log_dict['mode'],
                                                    log_dict['epoch'] - 1,
                                                    log_dict['iter'])
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = '{:.4f}'.format(val)
            log_items.append('{}: {}'.format(name, val))
        log_str += ', '.join(log_items)
        runner.logger.info(log_str)

    def _dump_log(self, log_dict, runner):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        # only append log at last line
        if runner.rank == 0:
            with open(self.json_log_path, 'a+') as f:
                dump(json_log, f, file_format='json')
                f.write('\n')

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def log(self, runner):
        log_dict = OrderedDict()
        # training mode if the output contains the key "time"
        mode = runner.mode
        log_dict['mode'] = mode
        log_dict['epoch'] = runner.epoch + 1
        log_dict['iter'] = runner.inner_iter + 1
        # only record lr of the first param group
        log_dict['lr'] = runner.current_lr.numpy()
        log_dict.update(runner.log_buffer.output)
        if runner.rank == 0:
            self._log_info(log_dict, runner)
            self._dump_log(log_dict, runner)
            
@HOOKS.register("TextLoggerHook")
def build_text_logger_hook(cfg):
    return TextLoggerHook(interval=cfg.LOG_INTERVAL)
