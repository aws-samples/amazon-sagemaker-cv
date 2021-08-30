# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
import sys
import logging
import os
import time
import numpy as np
import collections
from .log_buffer import LogBuffer
from sagemakercv.utils import comm
from .hooks import CheckpointHook, Hook, IterTimerHook
from .priority import get_priority
import torch

class Runner(object):
    """A training helper.
    Args:
        model (:obj:`tf.keras.Model`): The model to be run.
        train_step (callable): A callable method that process a data
            batch. The interface of this method should be
            `train_step(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`keras.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """
    def __init__(self,
                 model,
                 trainer,
                 cfg,
                 device,
                 optimizer=None,
                 scheduler=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        assert callable(trainer)
        self.model = model
        self.optimizer = optimizer
        self.trainer = trainer
        self.cfg = cfg
        self.device = device 
        self.scheduler = scheduler
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if isinstance(work_dir, str):
            self.work_dir = os.path.abspath(work_dir)
        else:
            self.work_dir = cfg.OUTPUT_DIR
        os.makedirs(self.work_dir, exist_ok=True)
        self._model_name = self.model.__class__.__name__ 
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
    def rank(self):
        """
        Global rank of current process. (distributed training)
        """
        return comm.get_rank()

    @property
    def world_size(self):
        """
        Number of processes participating in the job.
        (distributed training)
        """
        return comm.get_world_size()

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
        return self.cfg.SOLVER.MAX_ITER
        
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
        logger.addHandler(file_handler)
        return logger

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
            level=level,
            stream=sys.stdout)
        logger = logging.getLogger(__name__)
        # TODO: This is doubling up output
        logger.addHandler(logging.StreamHandler(sys.stdout))
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = os.path.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
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
    
    @property
    def current_lr(self):
        return self.scheduler.get_lr()[0]
    
    def train_epoch(self, data_iterator):
        self.mode = 'train'
        self.model.train()
        self.call_hook('before_train_epoch')
        self._inner_iter = 0
        for iteration in range(self.num_examples):
            images, targets = next(data_iterator)
            self.call_hook('before_train_iter')
            self.losses = self.trainer(images, 
                                       targets, 
                                       self.model, 
                                       self.optimizer, 
                                       self.scheduler, 
                                       self.device,
                                       self.cfg.DTYPE,
                                       grad_clip=self.cfg.SOLVER.GRADIENT_CLIPPING)
            if self.rank == 0:
                self.log_buffer.update(self.losses, self._iter)
            self.call_hook('after_train_iter')
            self._iter += 1
            self._inner_iter += 1
            if self._iter >= self.cfg.SOLVER.MAX_ITER:
                break
        self._epoch += 1
        self.call_hook('after_train_epoch')
    
    def run(self, data_iterator, num_examples):
        epochs = self.cfg.SOLVER.MAX_ITER//num_examples + 1
        self.num_examples = num_examples
        if self.rank==0:
            self.logger.info('Start running, work_dir: %s', self.work_dir)
            self.logger.info('max: %d epochs', epochs)
        self.call_hook('before_run')
        while self._epoch < epochs:
            self.train_epoch(data_iterator)
        self.call_hook('after_run')
