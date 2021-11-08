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
import math 

DEBUG_LOSS_IMPLEMENTATION = False


if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    from tensorflow.python.keras.utils import losses_utils
    ReductionV2 = losses_utils.ReductionV2
else:
    ReductionV2 = tf.keras.losses.Reduction

def _calculate_giou(b1, b2, mode="giou"):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou


def _calculate_ciou(b1, b2, mode="diou"):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['diou', 'ciou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)

    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height

    if mode == "giou":
        giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
        return giou

    # CIoU - https://arxiv.org/pdf/1911.08287.pdf
    diag_length = tf.linalg.norm([enclose_height, enclose_width]) # tf.math.square(enclose_width) + tf.math.square(enclose_height)
    b1_center = tf.stack([(b1_ymin + b1_ymax) / 2., (b1_xmin + b1_xmax) / 2.])
    b2_center = tf.stack([(b2_ymin + b2_ymax) / 2., (b2_xmin + b2_xmax) / 2.])
    centers_dist = tf.linalg.norm([b1_center-b2_center])

    diou = iou - tf.math.divide_no_nan(centers_dist**2, diag_length**2)

    if mode == "diou":
        return diou

    arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(tf.math.divide_no_nan(b2_width, b2_height))
    v = 4.0 * ((arctan / math.pi) ** 2)

    # apply aspect ratio penalty only if IoU > 0.5 and GT box is in medium or large category
    # aspect_penalty_mask = tf.cast(tf.math.logical_and(iou > 0.5, b1_area > 1024.), b1.dtype) # don't know size of box after resize!
    aspect_penalty_mask = tf.cast(iou > 0.5, b1.dtype)
    alpha = aspect_penalty_mask * tf.math.divide_no_nan(v, 1.0 - iou + v)

    ciou = diou - alpha * v

    return ciou



def _giou_loss(y_true, y_pred, weights, reduction='sum'):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, tf.float32)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])
    weights = tf.reshape(weights, [-1, 4])
    giou = _calculate_giou(y_true, y_pred)
    giou_loss = 1. - giou
    giou_loss = tf.tile(tf.expand_dims(giou_loss, -1), [1, 4]) * weights # only take pos example contributions
    avg_factor = tf.math.count_nonzero(weights, dtype=tf.float32)
    if reduction == 'sum':
        giou_loss = tf.math.divide_no_nan(tf.math.reduce_sum(giou_loss), avg_factor)
    else:
        giou_loss = tf.math.divide_no_nan(giou_loss, avg_factor)
    assert giou_loss.dtype == tf.float32
    return giou_loss


def _ciou_loss(y_true, y_pred, weights):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, tf.float32)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])
    weights = tf.reshape(weights, [-1, 4])
    ciou = _calculate_ciou(b1=y_true, b2=y_pred, mode="ciou")
    ciou_loss = 1. - ciou

    ciou_loss = tf.tile(tf.expand_dims(ciou_loss, -1), [1, 4]) * weights # only take pos example contributions
    ciou_loss = tf.math.divide_no_nan(tf.math.reduce_sum(ciou_loss), tf.math.count_nonzero(weights, dtype=tf.float32))
    assert ciou_loss.dtype == tf.float32
    return ciou_loss


def _l1_loss(y_true, y_pred, weights, delta=0.0):
    l1_loss = tf.compat.v1.losses.absolute_difference(y_true, y_pred, weights=weights)
    assert l1_loss.dtype == tf.float32
    DEBUG_LOSS_IMPLEMENTATION = False
    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.huber_loss(
            y_true,
            y_pred,
            weights=weights,
            delta=delta,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        print_op = tf.print("Huber Loss - MLPerf:", mlperf_loss, " && Legacy Loss:", l1_loss)

        with tf.control_dependencies([print_op]):
            l1_loss = tf.identity(l1_loss)

    return l1_loss


def _huber_loss(y_true, y_pred, weights, delta, reduction=ReductionV2.SUM):

    num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)

    huber_keras_loss = tf.keras.losses.Huber(
        delta=delta,
        reduction=reduction,
        name='huber_loss'
    )

    if LooseVersion(tf.__version__) >= LooseVersion("2.2.0"):
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)

    huber_loss = huber_keras_loss(
        y_true,
        y_pred,
        sample_weight=weights
    )

    assert huber_loss.dtype == tf.float32

    huber_loss = tf.math.divide_no_nan(huber_loss, num_non_zeros, name="huber_loss")

    assert huber_loss.dtype == tf.float32

    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.huber_loss(
            y_true,
            y_pred,
            weights=weights,
            delta=delta,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        print_op = tf.print("Huber Loss - MLPerf:", mlperf_loss, " && Legacy Loss:", huber_loss)

        with tf.control_dependencies([print_op]):
            huber_loss = tf.identity(huber_loss)

    return huber_loss


def _sigmoid_cross_entropy(multi_class_labels, logits, weights, sum_by_non_zeros_weights=False, label_smoothing=0.0):

    assert weights.dtype == tf.float32

    sigmoid_cross_entropy = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=multi_class_labels,
        logits=logits,
        label_smoothing=label_smoothing,
        reduction=tf.compat.v1.losses.Reduction.NONE
    )

    assert sigmoid_cross_entropy.dtype == tf.float32

    sigmoid_cross_entropy = tf.math.multiply(sigmoid_cross_entropy, weights)
    sigmoid_cross_entropy = tf.math.reduce_sum(sigmoid_cross_entropy)

    assert sigmoid_cross_entropy.dtype == tf.float32

    if sum_by_non_zeros_weights:
        num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)
        sigmoid_cross_entropy = tf.math.divide_no_nan(
            sigmoid_cross_entropy,
            num_non_zeros,
            name="sum_by_non_zeros_weights"
        )

    assert sigmoid_cross_entropy.dtype == tf.float32

    if DEBUG_LOSS_IMPLEMENTATION:

        if sum_by_non_zeros_weights:
            reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

        else:
            reduction = tf.compat.v1.losses.Reduction.SUM

        mlperf_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=multi_class_labels,
            logits=logits,
            weights=weights,
            reduction=reduction
        )

        print_op = tf.print(
            "Sigmoid X-Entropy Loss (%s) - MLPerf:" % reduction, mlperf_loss, " && Legacy Loss:", sigmoid_cross_entropy
        )

        with tf.control_dependencies([print_op]):
            sigmoid_cross_entropy = tf.identity(sigmoid_cross_entropy)

    return sigmoid_cross_entropy


def _softmax_cross_entropy(onehot_labels, logits, label_smoothing=0.0):

    num_non_zeros = tf.math.count_nonzero(onehot_labels, dtype=tf.float32)
    if label_smoothing == 0.0:
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=onehot_labels,
            logits=logits
        )
    else:
        softmax_cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels,
            logits,
            label_smoothing=label_smoothing,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

    assert softmax_cross_entropy.dtype == tf.float32

    if label_smoothing == 0.0:
        softmax_cross_entropy = tf.math.reduce_sum(softmax_cross_entropy)
        softmax_cross_entropy = tf.math.divide_no_nan(softmax_cross_entropy, num_non_zeros, name="softmax_cross_entropy")

    assert softmax_cross_entropy.dtype == tf.float32

    DEBUG_LOSS_IMPLEMENTATION = False

    if DEBUG_LOSS_IMPLEMENTATION:

        mlperf_loss = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        print_op = tf.print("Softmax X-Entropy Loss - MLPerf:", mlperf_loss, " && Legacy Loss:", softmax_cross_entropy)

        with tf.control_dependencies([print_op]):
            softmax_cross_entropy = tf.identity(softmax_cross_entropy)

    return softmax_cross_entropy

def _fast_rcnn_box_carl_loss(box_outputs, box_targets, class_targets,
            class_outputs, beta=0.2, gamma=1.0, num_classes=91,
            loss_type='huber', normalizer=1.0, delta=1.):
    """Computes classification aware box regression loss."""

    with tf.name_scope('fast_rcnn_carl_loss'):
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        if loss_type == 'huber':
            box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta, reduction=ReductionV2.NONE)
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


def mask_rcnn_loss(mask_outputs, mask_targets, select_class_targets, params):
    """Computes the mask loss of Mask-RCNN.
    This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`. Note that the selection logic is
    done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.
    Args:
    mask_outputs: a float tensor representing the prediction for each mask,
      with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    mask_targets: a float tensor representing the binary mask of ground truth
      labels for each mask with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    select_class_targets: a tensor with a shape of [batch_size, num_masks],
      representing the foreground mask targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
    Returns:
    mask_loss: a float tensor representing total mask loss.
    """
    with tf.name_scope('mask_loss'):
        batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()

        weights = tf.tile(
            tf.reshape(tf.greater(select_class_targets, 0), [batch_size, num_masks, 1, 1]),
            [1, 1, mask_height, mask_width]
        )
        weights = tf.cast(weights, tf.float32)

        loss = _sigmoid_cross_entropy(
            multi_class_labels=mask_targets,
            logits=mask_outputs,
            weights=weights,
            sum_by_non_zeros_weights=True,
            label_smoothing=params['label_smoothing']
        )

        mrcnn_loss = params['mrcnn_weight_loss_mask'] * loss

        return mrcnn_loss
