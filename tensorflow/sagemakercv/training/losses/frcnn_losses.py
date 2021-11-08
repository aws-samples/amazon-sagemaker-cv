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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion

import tensorflow as tf
from sagemakercv.core import box_utils
from .losses import _huber_loss, _l1_loss, _giou_loss, _ciou_loss, _softmax_cross_entropy, _sigmoid_cross_entropy

DEBUG_LOSS_IMPLEMENTATION = False


if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    from tensorflow.python.keras.utils import losses_utils
    ReductionV2 = losses_utils.ReductionV2
else:
    ReductionV2 = tf.keras.losses.Reduction

class FastRCNNLoss(object):

    def __init__(self, 
                 num_classes=91,
                 box_loss_type='huber',
                 use_carl_loss=False,
                 bbox_reg_weights=(10., 10., 5., 5.),
                 fast_rcnn_box_loss_weight=1.,
                 image_size=(832., 1344.),
                 class_agnostic_box=False):
        self.num_classes = num_classes
        self.box_loss_type = box_loss_type
        self.use_carl_loss = use_carl_loss
        self.bbox_reg_weights = bbox_reg_weights
        self.fast_rcnn_box_loss_weight = fast_rcnn_box_loss_weight
        self.image_size = image_size
        self.class_agnostic_box = class_agnostic_box
    
    def _fast_rcnn_class_loss(self, 
                              class_outputs, 
                              class_targets_one_hot, 
                              normalizer=1.0):
        """Computes classification loss."""
        with tf.name_scope('fast_rcnn_class_loss'):
            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.
            
            if self.num_classes==1:
                class_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
                            multi_class_labels=class_targets_one_hot,
                            logits=class_outputs,
                            label_smoothing=0.0,
                            reduction=tf.compat.v1.losses.Reduction.MEAN
                        )
            else:
                class_loss = _softmax_cross_entropy(onehot_labels=class_targets_one_hot, logits=class_outputs)

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                class_loss /= normalizer

        return class_loss
    
    def _fast_rcnn_box_loss(self, 
                            box_outputs, 
                            box_targets, 
                            class_targets, 
                            loss_type='huber', 
                            normalizer=1.0, 
                            delta=1.):
        """Computes box regression loss."""
        # delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

        with tf.name_scope('fast_rcnn_box_loss'):
            mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.
            if loss_type == 'huber':
                box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
            elif loss_type == 'giou':
                box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
            elif loss_type == 'ciou':
                box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
            elif loss_type == 'l1_loss':
                box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
            else:
                # box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
                raise NotImplementedError

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                box_loss /= normalizer

        return box_loss
    
    def _fast_rcnn_box_carl_loss(self, 
                                 box_outputs, 
                                 box_targets, 
                                 class_targets,
                                 class_outputs, 
                                 beta=0.2, 
                                 gamma=1.0, 
                                 num_classes=91,
                                 loss_type='huber', 
                                 normalizer=1.0, 
                                 delta=1.):
        """Computes classification aware box regression loss."""

        with tf.name_scope('fast_rcnn_carl_loss'):
            mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.
            if loss_type == 'huber':
                box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, 
                                       delta=delta, reduction=ReductionV2.NONE)
            elif loss_type == 'giou':
                box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, reduction='none')
                box_loss = tf.reshape(box_loss, [-1, 512, 4]) # FIXME
            elif loss_type == 'ciou':
                box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
            else:
                raise NotImplementedError

            oh_targets = tf.one_hot(class_targets, depth=num_classes)
            class_scores = tf.nn.softmax(class_outputs, axis=-1)
            top_class_score = class_scores * oh_targets
            pos_cls_score = top_class_score[:,:,1:] # ignore GT score
            pos_cls_score = tf.reduce_max(pos_cls_score, axis=-1)
            # carl_loss_weights = tf.pow(beta + (1. - beta) * pos_cls_score, gamma)
            # since gamma is 1.0 right now just use the linear combination term
            carl_loss_weights = beta + (1. - beta) * pos_cls_score
            # zero out bias contributions from zero pos_cls_scores (GTs)
            carl_loss_weights = tf.where(pos_cls_score > 0.0, carl_loss_weights, tf.zeros_like(carl_loss_weights))
            # normalize carl_loss_weight to make its sum equal to num positive
            num_pos = tf.math.count_nonzero(class_targets, dtype=carl_loss_weights.dtype)
            weight_ratio = tf.math.divide_no_nan(num_pos, tf.reduce_sum(carl_loss_weights))
            carl_loss_weights *= weight_ratio

            loss_carl = tf.reduce_sum(box_loss * tf.expand_dims(carl_loss_weights, -1))
            loss_bbox = tf.reduce_sum(box_loss)

            assert loss_carl.dtype == tf.float32

            regression_loss = loss_carl + loss_bbox

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                regression_loss /= normalizer
            return regression_loss
    
    def __call__(self, 
                 class_outputs, 
                 box_outputs, 
                 class_targets, 
                 box_targets, 
                 rpn_box_rois, 
                 image_info):
        with tf.name_scope('fast_rcnn_loss'):
            class_targets = tf.cast(class_targets, dtype=tf.int32)
            mask_targets = class_targets
            if self.class_agnostic_box:
                mask_targets = tf.clip_by_value(class_targets, 0, 1)
            # Selects the box from `box_outputs` based on `class_targets`, with which
            # the box has the maximum overlap.
            batch_size, num_rois, _ = box_outputs.get_shape().as_list()
            box_num_classes = 2 if self.class_agnostic_box else self.num_classes
            box_outputs = tf.reshape(box_outputs, [batch_size, num_rois, box_num_classes, 4])

            box_indices = tf.reshape(
                mask_targets +
                tf.tile(tf.expand_dims(tf.range(batch_size) * num_rois * \
                                       box_num_classes, 1), [1, num_rois]) +
                tf.tile(tf.expand_dims(tf.range(num_rois) * box_num_classes, 0), [batch_size, 1]),
                [-1]
            )
            box_outputs = tf.matmul(
                tf.one_hot(
                    box_indices,
                    batch_size * num_rois * box_num_classes,
                    dtype=box_outputs.dtype
                ),
                tf.reshape(box_outputs, [-1, 4])
            )
            if self.box_loss_type in ['giou', 'ciou']:
                # decode outputs to move deltas back to coordinate space
                rpn_box_rois = tf.reshape(rpn_box_rois, [-1, 4])
                box_outputs = box_utils.decode_boxes(encoded_boxes=box_outputs, 
                                                     anchors=rpn_box_rois, 
                                                     weights=self.bbox_reg_weights)
                box_outputs = box_utils.clip_boxes(box_outputs, self.image_size)
            
            box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])
            if not self.use_carl_loss:
                box_loss = self._fast_rcnn_box_loss(
                    box_outputs=box_outputs,
                    box_targets=box_targets,
                    class_targets=mask_targets,
                    loss_type=self.box_loss_type,
                    normalizer=1.0
                )
            else:
                if self.class_agnostic_box:
                    raise NotImplementedError
                box_loss = self._fast_rcnn_box_carl_loss(
                    box_outputs=box_outputs,
                    box_targets=box_targets,
                    class_targets=mask_targets,
                    class_outputs=class_outputs,
                    loss_type=self.box_loss_type,
                    normalizer=2.0
                )
            
            box_loss *= self.fast_rcnn_box_loss_weight
            
            use_sparse_x_entropy = False
            
            _class_targets = class_targets \
                if use_sparse_x_entropy \
                else tf.one_hot(class_targets, self.num_classes)
            
            class_loss = self._fast_rcnn_class_loss(
                class_outputs=class_outputs,
                class_targets_one_hot=_class_targets,
                normalizer=1.0
            )
            
            total_loss = class_loss + box_loss
            
        return total_loss, class_loss, box_loss

class MaskRCNNLoss(object):
    def __init__(self,
                 mrcnn_weight_loss_mask=1.,
                 label_smoothing=0.0):
        self.mrcnn_weight_loss_mask = mrcnn_weight_loss_mask
        self.label_smoothing = label_smoothing
    
    def __call__(self, mask_outputs, mask_targets, select_class_targets):
        with tf.name_scope('mask_loss'):
            batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()
            weights = tf.tile(
                tf.reshape(tf.greater(select_class_targets, 0), 
                           [batch_size, num_masks, 1, 1]),
                [1, 1, mask_height, mask_width]
            )
            weights = tf.cast(weights, tf.float32)
            
            loss = _sigmoid_cross_entropy(
                multi_class_labels=mask_targets,
                logits=mask_outputs,
                weights=weights,
                sum_by_non_zeros_weights=True,
                label_smoothing=self.label_smoothing
            )
            
            mrcnn_loss = self.mrcnn_weight_loss_mask * loss
            
        return mrcnn_loss