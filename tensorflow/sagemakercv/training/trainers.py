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

import logging
import tensorflow as tf
from sagemakercv.utils.dist_utils import MPI_rank_and_size, MPI_is_distributed, MPI_rank
from .builder import TRAINERS
from sagemakercv.utils.logging_formatter import logging

class DetectionTrainer(object):
    """
    Training loop for object detection models
    """
    
    def __init__(self,
                 model,
                 optimizer,
                 dist=None,
                 fp16=False,
                 global_gradient_clip_ratio=0.0,
                 weight_decay=0.0):
        self.model = model
        self.optimizer = optimizer
        self.dist = self.initialize_dist(dist)
        self.fp16 = fp16
        self.global_gradient_clip_ratio = global_gradient_clip_ratio
        self.weight_decay = weight_decay
    
    @tf.function  
    def __call__(self, data_batch, training=True, broadcast=False):
        if not training:
            # TODO:
            # this only works with the dict output from val change data pipeline
            # to make training and val match
            model_outputs = self.model(data_batch['features'], data_batch.get('labels'), training=training)
            model_outputs.update({
                'source_ids': data_batch['features']['source_ids'],
                'image_info': data_batch['features']['image_info'],
            })
            return model_outputs
        else:
            with tf.GradientTape() as tape:
                model_outputs = self.model(*data_batch, training=True, weight_decay=self.weight_decay)
                if self.fp16:
                    scaled_loss = self.optimizer.get_scaled_loss(model_outputs['total_loss'])
            if self.dist!=None:
                tape = self.dist.DistributedGradientTape(tape)
            if self.fp16:
                scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(model_outputs['total_loss'], self.model.trainable_variables)
            if self.global_gradient_clip_ratio > 0.0:
                all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
                (clipped_grads, _) = tf.clip_by_global_norm(gradients, 
                                                    clip_norm=self.global_gradient_clip_ratio,
                                                    use_norm=tf.cond(all_are_finite, 
                                                        lambda: tf.linalg.global_norm(gradients), 
                                                        lambda: tf.constant(1.0)))
                gradients = clipped_grads
            grads_and_vars = []
            for grad, var in zip(gradients, self.model.trainable_variables):
                if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                    grad = 2.0 * grad
                grads_and_vars.append((grad, var))
            self.optimizer.apply_gradients(grads_and_vars)
            if self.dist!=None and broadcast:
                if MPI_rank()==0:
                    logging.info("Broadcasting model")
                self.dist.broadcast_variables(self.model.variables, 0)
                self.dist.broadcast_variables(self.optimizer.variables(), 0)
            losses = {i:j for i,j in model_outputs.items() if "loss" in i}
            model_outputs.update({
                'source_ids': data_batch[0]['source_ids'],
                'image_info': data_batch[0]['image_info'],
            })
            return losses, model_outputs
    
    def initialize_dist(self, dist):
        if dist is None:
            return
        if dist.lower() in ['hvd', 'horovod']:
            logging.info("Using Horovod For Distributed Training")
            import horovod.tensorflow as dist
            dist.init()
            return dist
        elif dist.lower() in ['smd', 'sagemaker', 'smddp']:
            logging.info("Using Sagemaker For Distributed Training")
            import smdistributed.dataparallel.tensorflow as dist
            dist.init()
            return dist
        else:
            raise NotImplementedError

@TRAINERS.register("DetectionTrainer")
def build_detection_trainer(cfg, model, optimizer, dist=None):
    return DetectionTrainer(model=model,
                            optimizer=optimizer,
                            dist=dist,
                            fp16=cfg.SOLVER.FP16,
                            global_gradient_clip_ratio=cfg.SOLVER.GRADIENT_CLIP_RATIO,
                            weight_decay=0.0 if cfg.SOLVER.OPTIMIZER in ["NovoGrad", "Adam"] \
                                             else cfg.SOLVER.WEIGHT_DECAY)
