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
from sagemakercv.training.losses import FastRCNNLoss

__all__ = ["BBoxHead"]

class StandardBBoxHead(tf.keras.Model):
    def __init__(self, 
                 num_classes=91, 
                 mlp_head_dim=1024, 
                 name="box_head", 
                 trainable=True,
                 class_agnostic_box=False,
                 loss_cfg=dict(num_classes=91,
                     box_loss_type='huber',
                     use_carl_loss=False,
                     bbox_reg_weights=(10., 10., 5., 5.),
                     fast_rcnn_box_loss_weight=1.,
                     image_size=(832., 1344.),
                     class_agnostic_box=False,
                    ),
                 *args, 
                 **kwargs):
        """Box and class branches for the Mask-RCNN model.
        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: a integer for the number of classes.
        mlp_head_dim: a integer that is the hidden dimension in the fully-connected
          layers.
        """
        super(StandardBBoxHead, self).__init__(name=name, trainable=trainable, *args, **kwargs)
        
        self._num_classes = num_classes
        self._mlp_head_dim = mlp_head_dim
        self._class_agnostic_box = class_agnostic_box
        
        self._bbox_dense_0 = tf.keras.layers.Dense(
                units=mlp_head_dim,
                activation=tf.nn.relu,
                trainable=trainable,
                name='bbox_dense_0'
            )
        
        self._bbox_dense_1 = tf.keras.layers.Dense(
                units=mlp_head_dim,
                activation=tf.nn.relu,
                trainable=trainable,
                name='bbox_dense_1'
            )
        
        self._dense_class = tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.keras.initializers.Zeros(),
                trainable=trainable,
                name='class-predict'
            )
        
        self._dense_box = tf.keras.layers.Dense(
                8 if self._class_agnostic_box else num_classes * 4,
                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer=tf.keras.initializers.Zeros(),
                trainable=trainable,
                name='box-predict'
            )
        self.loss = FastRCNNLoss(**loss_cfg)
        
        
    def call(self, inputs, **kwargs):
        """
        Returns:
        class_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes], representing the class predictions.
        box_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes * 4], representing the box predictions.
        box_features: a tensor with a shape of
          [batch_size, num_rois, mlp_head_dim], representing the box features.
        """

        # reshape inputs before FC.
        batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()

        net = tf.reshape(inputs, [batch_size, num_rois, height * width * filters])

        net = self._bbox_dense_0(net)

        box_features = self._bbox_dense_1(net)

        class_outputs = self._dense_class(box_features)

        box_outputs = self._dense_box(box_features)

        return class_outputs, box_outputs, box_features
    
@HEADS.register("StandardBBoxHead")
def build_standard_box_head(cfg):
    bbox_head = StandardBBoxHead
    if cfg.MODEL.RCNN.ROI_HEAD == "CascadeRoIHead":
        assert len(cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS)==len(cfg.MODEL.RCNN.CASCADE.THRESHOLDS)
        num_stages = len(cfg.MODEL.RCNN.CASCADE.THRESHOLDS)
        return [StandardBBoxHead(num_classes=cfg.INPUT.NUM_CLASSES,
                                 mlp_head_dim=cfg.MODEL.FRCNN.MLP_DIM,
                                 name=f"box_head_{stage}",
                                 trainable=cfg.MODEL.FRCNN.TRAINABLE,
                                 class_agnostic_box=True if stage<(num_stages-1) \
                                                    else cfg.MODEL.FRCNN.CLASS_AGNOSTIC,
                                 loss_cfg=dict(num_classes=cfg.INPUT.NUM_CLASSES,
                                         box_loss_type=cfg.MODEL.FRCNN.LOSS_TYPE,
                                         use_carl_loss=cfg.MODEL.FRCNN.CARL,
                                         bbox_reg_weights=cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS[stage],
                                         fast_rcnn_box_loss_weight=1,
                                         image_size=cfg.INPUT.IMAGE_SIZE,
                                         class_agnostic_box=True if stage<(num_stages-1) \
                                                    else cfg.MODEL.FRCNN.CLASS_AGNOSTIC,
                                  )) for stage in range(num_stages)]
    return bbox_head(num_classes=cfg.INPUT.NUM_CLASSES,
                     mlp_head_dim=cfg.MODEL.FRCNN.MLP_DIM,
                     trainable=cfg.MODEL.FRCNN.TRAINABLE,
                     class_agnostic_box=cfg.MODEL.FRCNN.CLASS_AGNOSTIC,
                     loss_cfg=dict(num_classes=cfg.INPUT.NUM_CLASSES,
                                   box_loss_type=cfg.MODEL.FRCNN.LOSS_TYPE,
                                   use_carl_loss=cfg.MODEL.FRCNN.CARL,
                                   bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS,
                                   fast_rcnn_box_loss_weight=cfg.MODEL.FRCNN.LOSS_WEIGHT,
                                   image_size=cfg.INPUT.IMAGE_SIZE,
                                   class_agnostic_box=cfg.MODEL.FRCNN.CLASS_AGNOSTIC,
                                  ))