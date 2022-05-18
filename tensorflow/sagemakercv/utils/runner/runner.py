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

import sys
import logging
import os
import time
from datetime import datetime
import numpy as np
import collections
from sagemakercv.utils.runner import LogBuffer
from sagemakercv.utils.dist_utils import get_dist_info, master_only
from .priority import get_priority
from .hooks import Hook
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

class Runner(object):
    """A training helper.
    Args:
        model (:obj:`tf.keras.Model`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`keras.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """
    def __init__(self,
                 trainer,
                 cfg,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None, 
                 name=None):
        self.trainer = trainer
        self.cfg = cfg
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if work_dir is not None:
            self.work_dir = os.path.abspath(work_dir)
        else:
            self.work_dir = self.cfg.PATHS.OUT_DIR
        os.makedirs(self.work_dir, exist_ok=True)
        self.init_tensorboard()
        if name is None:
            self._model_name = self.trainer.model.__class__.__name__ + " " + self.timestamp
        self._rank, self._local_rank, self._size, self._local_size = get_dist_info()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        
    @property
    def model_name(self):
        """
        Name of the model, usually the module class name.
        """
        return self._model_name

    @property
    def local_rank(self):
        """
        Local rank of current process
        """
        return self._local_rank

    @property
    def rank(self):
        """
        Global rank of current process. (distributed training)
        """
        return self._rank

    @property
    def world_size(self):
        """
        Number of processes participating in the job.
        (distributed training)
        """
        return self._world_size

    @property
    def local_size(self):
        """
        Number of processes running in the same node as this runner.
        (distributed training)
        """
        return self._local_size

    @property
    def hooks(self):
        """
        A list of registered hooks.
        """
        return self._hooks

    @property
    def epoch(self):
        """
        Current epoch.
        """
        return self._epoch

    @property
    def iter(self):
        """
        Current iteration
        """
        return self._iter

    @property
    def inner_iter(self):
        """
        Iteration in an epoch.
        """
        return self._inner_iter

    @property
    def max_iters(self):
        """
        Maximum training iterations.
        """
        return self.cfg.SOLVER.MAX_ITERS
    
    @property
    def train_batch_size(self):
        """
        Maximum training epochs.
        """
        return self.cfg.INPUT.TRAIN_BATCH_SIZE
    
    @property
    def train_batch_size_per_device(self):
        """
        Maximum training epochs.
        """
        return self.train_batch_size//self.world_size
    
    @property
    def steps_per_epoch(self):
        return self.cfg.SOLVER.NUM_IMAGES//self.train_batch_size
    
    @property
    def max_epochs(self):
        """
        Maximum training epochs.
        """
        return self.max_iters//self.steps_per_epoch + 1
    
    @property
    def current_lr(self):
        return self.trainer.optimizer.lr(self.trainer.optimizer.iterations)
        
    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        file_handler.setStream(sys.stdout)
        logger.addHandler(file_handler)
        return logger
    
    @master_only
    def init_tensorboard(self):
        self.tensorboard_dir = os.path.join(self.work_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def init_logger(self, log_dir=None, level=logging.INFO):
        """
        Init the logger.
        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.
        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', 
            level=level)
        logger = logging.getLogger(__name__)
        # logger = tf.get_logger()
        # TODO: This is doubling up output
        logger.addHandler(logging.StreamHandler(sys.stdout))
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = os.path.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        logger.setLevel(level)
        return logger
    
    def register_hook(self, hook, priority='NORMAL'):
        """
        Register a hook into the hook list.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
    
    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
    
    def train_epoch(self, dataset, steps=None, **kwargs):
        if self.rank==0:
            self.logger.info(f'Starting epoch: {self._epoch + 1} of {self.max_epochs}')
        if isinstance(dataset, dataset_ops._OptionsDataset):
            dataset=iter(dataset.repeat())
        if steps==None:
            steps = self.steps_per_epoch
        self.num_examples = steps
        self.mode = 'train'
        broadcast = True
        self.call_hook('before_train_epoch')
        self._inner_iter = 0
        for i in range(self.num_examples):
            self.data_batch = next(dataset)
            self.call_hook('before_train_iter')
            self.losses, self.outputs = self.trainer(self.data_batch, 
                                                     training=True, 
                                                     broadcast=broadcast)
            broadcast = False
            if not isinstance(self.losses, dict):
                raise TypeError('trainer must return a dict')
            if self.rank == 0:
                self.log_buffer.update(self.losses, self._iter)
            self.call_hook('after_train_iter')
            self._iter += 1
            self._inner_iter += 1
            if self._iter >= self.max_iters:
                break
        self._epoch += 1
        self.call_hook('after_train_epoch')
    
    def run(self, dataset):
        if isinstance(dataset, dataset_ops._OptionsDataset):
            dataset=iter(dataset.repeat())
        if self.rank==0:
            self.logger.info('Start running, work_dir: %s', self.work_dir)
            self.logger.info('max: %d epochs', self.max_epochs)
        self.call_hook('before_run')
        while self._epoch < self.max_epochs:
            self.train_epoch(dataset)
        self.call_hook('after_run')
