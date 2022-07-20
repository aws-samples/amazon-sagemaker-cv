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

from ...builder import HEADS
from sagemakercv.training.losses import MaskRCNNLoss

__all__ = ["StandardMaskHead"]

class StandardMaskHead(tf.keras.Model):

    @staticmethod
    def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
        """Returns the stddev of random normal initialization as MSRAFill."""
        # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463
        # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
        # stddev = (2/(3*3*256))^0.5 = 0.029
        return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

    def __init__(
            self,
            num_classes=91,
            mrcnn_resolution=28,
            is_gpu_inference=False,
            name="mask_head",
            trainable=True,
            loss_cfg=dict(
                mrcnn_weight_loss_mask=1.,
                label_smoothing=0.0,
            ),
            *args,
            **kwargs
    ):
        """Mask branch for the Mask-RCNN model.
        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: an integer for the number of classes.
        mrcnn_resolution: an integer that is the resolution of masks.
        is_gpu_inference: whether to build the model for GPU inference.
        """
        super(StandardMaskHead, self).__init__(name=name, trainable=trainable, *args, **kwargs)

        # self._class_indices = class_indices
        self._num_classes = num_classes
        self._mrcnn_resolution = mrcnn_resolution
        self._is_gpu_inference = is_gpu_inference

        self._conv_stage1 = list()
        kernel_size = (3, 3)
        fan_out = 256

        init_stddev = StandardMaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        for conv_id in range(4):
            self._conv_stage1.append(tf.keras.layers.Conv2D(
                fan_out,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding='same',
                dilation_rate=(1, 1),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
                bias_initializer=tf.keras.initializers.Zeros(),
                trainable=trainable,
                name='mask-conv-l%d' % conv_id
            ))

        kernel_size = (2, 2)
        fan_out = 256

        init_stddev = StandardMaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage2 = tf.keras.layers.Conv2DTranspose(
            fan_out,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable,
            name='conv5-mask'
        )

        kernel_size = (1, 1)
        fan_out = self._num_classes

        init_stddev = StandardMaskHead._get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)

        self._conv_stage3 = tf.keras.layers.Conv2D(
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='valid',
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=trainable,
            name='mask_fcn_logits'
        )
        
        self.loss = MaskRCNNLoss(**loss_cfg)

    def call(self, inputs, class_indices, **kwargs):
        """
        class_indices: a Tensor of shape [batch_size, num_rois], indicating
          which class the ROI is.
        Returns:
        mask_outputs: a tensor with a shape of
          [batch_size, num_masks, mask_height, mask_width],
          representing the mask predictions.
        fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
          representing the fg mask targets.
        Raises:
        ValueError: If boxes is not a rank-3 tensor or the last dimension of
          boxes is not 4.
        """

        batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()

        net = tf.reshape(inputs, [-1, height, width, filters])

        for conv_id in range(4):
            net = self._conv_stage1[conv_id](net)

        net = self._conv_stage2(net)

        mask_outputs = self._conv_stage3(net)

        mask_outputs = tf.reshape(
            mask_outputs,
            [-1, num_rois, self._mrcnn_resolution, self._mrcnn_resolution, self._num_classes]
        )

        with tf.name_scope('masks_post_processing'):

            mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])

            indices_dtype = tf.float32 if self._is_gpu_inference else tf.int32
            class_indices = tf.cast(class_indices, indices_dtype)

            if batch_size == 1:
                indices = tf.reshape(
                    tf.reshape(
                        tf.range(num_rois, dtype=indices_dtype),
                        [batch_size, num_rois, 1]
                    ) * self._num_classes + tf.expand_dims(class_indices, axis=-1),
                    [batch_size, -1]
                )
                indices = tf.cast(indices, tf.int32)

                mask_outputs = tf.gather(
                    tf.reshape(mask_outputs, [batch_size, -1, self._mrcnn_resolution, self._mrcnn_resolution]),
                    indices,
                    axis=1
                )

                mask_outputs = tf.squeeze(mask_outputs, axis=1)
                mask_outputs = tf.reshape(
                    mask_outputs,
                    [batch_size, num_rois, self._mrcnn_resolution, self._mrcnn_resolution])

            else:
                batch_indices = (
                        tf.expand_dims(tf.range(batch_size, dtype=indices_dtype), axis=1) *
                        tf.ones([1, num_rois], dtype=indices_dtype)
                )

                mask_indices = (
                        tf.expand_dims(tf.range(num_rois, dtype=indices_dtype), axis=0) *
                        tf.ones([batch_size, 1], dtype=indices_dtype)
                )

                gather_indices = tf.stack([batch_indices, mask_indices, class_indices], axis=2)

                if self._is_gpu_inference:
                    gather_indices = tf.cast(gather_indices, dtype=tf.int32)

                mask_outputs = tf.gather_nd(mask_outputs, gather_indices)

        return mask_outputs
    
@HEADS.register("StandardMaskHead")    
def build_standard_mask_head(cfg):
    mask_head = StandardMaskHead
    return mask_head(num_classes=cfg.INPUT.NUM_CLASSES,
                     mrcnn_resolution=cfg.MODEL.MRCNN.RESOLUTION,
                     is_gpu_inference=cfg.MODEL.MRCNN.GPU_INFERENCE,
                     trainable=cfg.MODEL.MRCNN.TRAINABLE,
                     loss_cfg=dict(
                                mrcnn_weight_loss_mask=cfg.MODEL.MRCNN.LOSS_WEIGHT,
                                label_smoothing=cfg.MODEL.MRCNN.LABEL_SMOOTHING,
                            ))