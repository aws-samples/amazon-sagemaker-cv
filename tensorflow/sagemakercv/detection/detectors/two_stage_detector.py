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

from ..builder import DETECTORS, build_backbone, build_dense_head, build_neck, build_roi_head

class TwoStageDetector(tf.keras.models.Model):
    def __init__(self,
                 backbone,
                 neck,
                 dense_head,
                 roi_head):
        super(TwoStageDetector, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = dense_head
        self.roi_head = roi_head

    def call(self, features, labels=None, training=True, weight_decay=0.0):
        x = self.backbone(features['images'], training=training)
        feature_maps = self.neck(x, training=training)
        scores_outputs, box_outputs, proposals = self.rpn_head(feature_maps,
                                                               features['image_info'],
                                                               training=training)
        model_outputs = {'images': features['images'], 'image_info': features['image_info']}
        if training:
            model_outputs.update(self.roi_head(feature_maps, features['image_info'], proposals[0],
                     gt_bboxes=labels['gt_boxes'], gt_labels=labels['gt_classes'],
                     gt_masks=labels.get('cropped_gt_masks', None), training=training))
            total_rpn_loss, rpn_score_loss, rpn_box_loss = self.rpn_head.loss(scores_outputs, box_outputs, labels)
            model_outputs.update({"total_rpn_loss": total_rpn_loss,
                                  "rpn_score_loss": rpn_score_loss,
                                  "rpn_box_loss": rpn_box_loss})
            loss_dict = self.parse_losses(model_outputs, weight_decay=weight_decay)
            model_outputs['total_loss'] = loss_dict['total_loss']
            if weight_decay>0.0:
                model_outputs['l2_loss'] = loss_dict['l2_loss']
        else:
            model_outputs.update(self.roi_head(feature_maps, features['image_info'], proposals[0], training=training))
        return model_outputs

    def parse_losses(self, losses, weight_decay=0.0):
        loss_dict = {i:j for i,j in losses.items() if "loss" in i and "total" not in i}
        if weight_decay>0.0:
            loss_dict['l2_loss'] = weight_decay * tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in self.trainable_variables
                        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
                    ])
        loss_dict['total_loss'] = sum(loss_dict.values())
        return loss_dict

    # overrided function for Keras model.compile
    def compile(self, loss, optimizer, run_eagerly=True):
        super(TwoStageDetector, self).compile(run_eagerly=run_eagerly)
        self.optimizer = optimizer
        self.loss = loss

    # overrided function for Keras model.fit
    @tf.function
    def train_step(self, data_batch):
        features, labels = data_batch
        with tf.GradientTape() as tape:
            model_outputs = self(features, labels, training=True)

        gradients = tape.gradient(model_outputs['total_loss'], self.trainable_variables)
        grads_and_vars = []
        for grad, var in zip(gradients, self.trainable_variables):
            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad
            grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)

        losses = {i: j for i, j in model_outputs.items() if "loss" in i}
        model_outputs.update({
            'source_ids': data_batch[0]['source_ids'],
            'image_info': data_batch[0]['image_info'],
        })
        return losses

@DETECTORS.register("TwoStageDetector")
def build_two_stage_detector(cfg):
    detector = TwoStageDetector
    return detector(backbone=build_backbone(cfg),
                    neck=build_neck(cfg),
                    dense_head=build_dense_head(cfg),
                    roi_head=build_roi_head(cfg))
