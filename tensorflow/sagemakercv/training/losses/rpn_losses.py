#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

"""Losses used for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion

import tensorflow as tf
from sagemakercv.core import box_utils
from .losses import _sigmoid_cross_entropy, _huber_loss, _giou_loss, _ciou_loss, _l1_loss

DEBUG_LOSS_IMPLEMENTATION = False


if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    from tensorflow.python.keras.utils import losses_utils
    ReductionV2 = losses_utils.ReductionV2
else:
    ReductionV2 = tf.keras.losses.Reduction

class RPNLoss(object):

    def __init__(self, min_level=2,
                       max_level=6,
                       box_loss_type='huber',
                       train_batch_size_per_gpu=1,
                       rpn_batch_size_per_im=256,
                       label_smoothing=0.0,
                       rpn_box_loss_weight=1.0):
        self.min_level = min_level
        self.max_level = max_level
        self.box_loss_type = box_loss_type
        self.train_batch_size_per_gpu = train_batch_size_per_gpu
        self.rpn_batch_size_per_im = rpn_batch_size_per_im
        self.label_smoothing = label_smoothing
        self.rpn_box_loss_weight = rpn_box_loss_weight

    def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0, delta=1. / 9):
        """Computes box regression loss."""
        # delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

        with tf.name_scope('rpn_box_loss'):
            mask = tf.not_equal(box_targets, 0.0)
            mask = tf.cast(mask, tf.float32)

            assert mask.dtype == tf.float32

            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.
            if self.box_loss_type == 'huber':
                box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
                # TODO: Test giou loss for rpn head
                '''elif self.box_loss_type == 'giou':
                    box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
                elif self.box_loss_type == 'ciou':
                    box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)'''
            elif self.box_loss_type == 'l1_loss':
                box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
            else:
                # box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
                raise NotImplementedError
            # box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)

            assert box_loss.dtype == tf.float32

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                box_loss /= normalizer

            assert box_loss.dtype == tf.float32

        return box_loss

    def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
        """Computes score loss."""

        with tf.name_scope('rpn_score_loss'):

            # score_targets has three values:
            # * (1) score_targets[i]=1, the anchor is a positive sample.
            # * (2) score_targets[i]=0, negative.
            # * (3) score_targets[i]=-1, the anchor is don't care (ignore).

            mask = tf.math.greater_equal(score_targets, 0)
            mask = tf.cast(mask, dtype=tf.float32)

            score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
            score_targets = tf.cast(score_targets, dtype=tf.float32)

            assert score_outputs.dtype == tf.float32
            assert score_targets.dtype == tf.float32

            score_loss = _sigmoid_cross_entropy(
                multi_class_labels=score_targets,
                logits=score_outputs,
                weights=mask,
                sum_by_non_zeros_weights=False,
                label_smoothing=self.label_smoothing
            )

            assert score_loss.dtype == tf.float32

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                score_loss /= normalizer

            assert score_loss.dtype == tf.float32

        return score_loss

    def __call__(self, score_outputs, box_outputs, labels):
        with tf.name_scope('rpn_loss'):
            score_losses = []
            box_losses = []
            for level in range(int(self.min_level), int(self.max_level + 1)):
                score_targets_at_level = labels['score_targets_%d' % level]
                box_targets_at_level = labels['box_targets_%d' % level]
                score_losses.append(
                    self._rpn_score_loss(
                        score_outputs=score_outputs[level],
                        score_targets=score_targets_at_level,
                        normalizer=tf.cast(self.train_batch_size_per_gpu \
                                           * self.rpn_batch_size_per_im, dtype=tf.float32)
                    ))
                box_losses.append(self._rpn_box_loss(
                    box_outputs=box_outputs[level],
                    box_targets=box_targets_at_level,
                    normalizer=1.0
                ))
            # Sum per level losses to total loss.
            rpn_score_loss = tf.add_n(score_losses)
            rpn_box_loss = self.rpn_box_loss_weight * tf.add_n(box_losses)
            total_rpn_loss = rpn_score_loss + rpn_box_loss
        return total_rpn_loss, rpn_score_loss, rpn_box_loss
