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
from time import sleep
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from sagemakercv.utils.dist_utils import master_only, is_sm_dist
from sagemakercv.data.coco import evaluation
from sagemakercv.utils import dist_utils
import numpy as np
from .hook import Hook
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
from ..builder import HOOKS
from sagemakercv.data import build_dataset

class CocoEvaluator(Hook):
    
    def __init__(self, 
                 eval_dataset,
                 annotations_file, 
                 per_epoch=True, 
                 tensorboard=True, 
                 include_mask_head=True, 
                 verbose=False):
        self.eval_dataset = eval_dataset
        self.annotations_file = annotations_file
        self.per_epoch = per_epoch
        if is_sm_dist():
            from smdistributed.dataparallel.tensorflow import get_worker_comm
            self.comm = get_worker_comm()
        else:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
        self.tensorboard = tensorboard
        self.verbose = verbose
        self.include_mask_head = include_mask_head
    
    def before_run(self, runner):
        self.eval_running = False
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count()//dist_utils.MPI_local_size())
        self.threads = []
        
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, 25) and self.eval_running:
            if self.threads_done():
                imgIds, box_predictions, mask_predictions = self.format_threads()
                imgIds_mpi_list = self.comm.gather(imgIds, root=0)
                box_predictions_mpi_list = self.comm.gather(box_predictions, root=0)
                mask_predictions_mpi_list = self.comm.gather(mask_predictions, root=0)
                if dist_utils.MPI_rank() == 0:
                    self.evaluate_thread = self.thread_pool.submit(self.evaluate,
                                                                   imgIds_mpi_list,
                                                                   box_predictions_mpi_list, 
                                                                   mask_predictions_mpi_list,
                                                                   runner.logger,
                                                                   runner.iter,
                                                                   runner.tensorboard_dir)
                self.eval_running = False
                self.comm.Barrier()
                
    def after_run(self, runner):
        if not self.per_epoch:
            self.eval_running = True
            self.threads = []
            if dist_utils.MPI_rank() == 0:
                runner.logger.info("Running eval for epoch {}".format(runner.epoch))
            for i, data in enumerate(self.eval_dataset):
                prediction = runner.trainer(data, training=False)
                prediction = {i:j.numpy() for i,j in prediction.items()}
                self.threads.append(self.thread_pool.submit(evaluation.process_prediction,
                                                            prediction))
            self.comm.Barrier()
        if dist_utils.MPI_rank() == 0:
            runner.logger.info("Processing final eval")
        while not self.threads_done():
            sleep(1)
        imgIds, box_predictions, mask_predictions = self.format_threads()
        imgIds_mpi_list = self.comm.gather(imgIds, root=0)
        box_predictions_mpi_list = self.comm.gather(box_predictions, root=0)
        mask_predictions_mpi_list = self.comm.gather(mask_predictions, root=0)
        if dist_utils.MPI_rank() == 0:
            self.evaluate(imgIds_mpi_list,
                          box_predictions_mpi_list, 
                          mask_predictions_mpi_list,
                          runner.logger,
                          runner.iter,
                          runner.tensorboard_dir)
        self.eval_running = False
        self.comm.Barrier()
        
    def after_train_epoch(self, runner):
        if self.per_epoch:
            self.eval_running = True
            self.threads = []
            if dist_utils.MPI_rank() == 0:
                runner.logger.info("Running eval for epoch {}".format(runner.epoch))
            for i, data in enumerate(self.eval_dataset):
                prediction = runner.trainer(data, training=False)
                prediction = {i:j.numpy() for i,j in prediction.items()}
                self.threads.append(self.thread_pool.submit(evaluation.process_prediction,
                                                            prediction))
            self.comm.Barrier()
    
    def threads_done(self):
        local_done = all([i.done() for i in self.threads])
        all_done = self.comm.allgather(local_done)
        return all(all_done)
    
    def format_threads(self):
        imgIds = []
        box_predictions = []
        mask_predictions = []
        for a_thread in self.threads:
            imgIds.extend(a_thread.result()[0])
            box_predictions.extend(a_thread.result()[1])
            mask_predictions.extend(a_thread.result()[2])
        return imgIds, box_predictions, mask_predictions
    
    def tensorboard_writer(self, stat_dict, iteration, tensorboard_dir):
        writer = tf.summary.create_file_writer(tensorboard_dir)
        with writer.as_default():
            for iou, metric_dict in stat_dict.items():
                for metric_name, metric in metric_dict.items():
                    tag = '{}/{}/{}'.format('eval', iou, metric_name)
                    tf.summary.scalar(tag, metric, step=iteration)
        writer.close()
        
    def log_results(self, 
                    stat_dict, 
                    logger):
        for iou, iou_dict in stat_dict.items():
            for stat, value in iou_dict.items():
                logger.info(f"{iou} {stat}: {value}")
                
    
    @master_only
    def evaluate(self, 
                 imgIds_mpi_list, 
                 box_predictions_mpi_list, 
                 mask_predictions_mpi_list,
                 logger,
                 iteration,
                 tensorboard_dir=None):
        imgIds = []
        box_predictions = []
        mask_predictions = []

        for i in imgIds_mpi_list:
            imgIds.extend(i)

        for i in box_predictions_mpi_list:
            box_predictions.extend(i)
        predictions = {'bbox': box_predictions}

        if self.include_mask_head:
            for i in mask_predictions_mpi_list:
                mask_predictions.extend(i)
            predictions['segm'] = mask_predictions

        logger.info("Running Evaluation for {} images". format(len(set(imgIds))))

        stat_dict = evaluation.evaluate_coco_predictions(self.annotations_file,
                                                         predictions.keys(),
                                                         predictions,
                                                         self.verbose)
        logger.info(f"{stat_dict}")
        self.log_results(stat_dict, logger)
        if self.tensorboard:
            self.tensorboard_writer(stat_dict, iteration, tensorboard_dir)
        
@HOOKS.register("CocoEvaluator")
def build_coco_evaluator(cfg):
    assert Path(cfg.PATHS.VAL_ANNOTATIONS).exists()
    return CocoEvaluator(build_dataset(cfg, mode='eval'),
                         cfg.PATHS.VAL_ANNOTATIONS,
                         per_epoch=cfg.SOLVER.EVAL_EPOCH_EVAL,
                         include_mask_head=cfg.MODEL.INCLUDE_MASK)
