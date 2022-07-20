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
from .anchor_head import AnchorHead
from sagemakercv.training.losses import RPNLoss
from sagemakercv.utils.dist_utils import MPI_size
from ..builder import HEADS

class StandardRPNHead(AnchorHead):
    def __init__(self, 
                 rpn_loss_cfg=dict(
                       min_level=2,
                       max_level=6,
                       box_loss_type='huber',
                       train_batch_size_per_gpu=1,
                       rpn_batch_size_per_im=256,
                       label_smoothing=0.0,
                       rpn_box_loss_weight=1.0
                 ),
                 *args,
                 **kwargs):
        super(StandardRPNHead, self).__init__(*args, **kwargs)
        self.loss = RPNLoss(**rpn_loss_cfg)
    
    def _init_layers(self):
        self.rpn_conv = tf.keras.layers.Conv2D(
                            self.feat_channels,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation=tf.nn.relu,
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                            padding='same',
                            trainable=self.trainable,
                            name='rpn'
                        )
        self.conv_cls = tf.keras.layers.Conv2D(
                            len(self.anchor_generator.aspect_ratios * \
                                self.anchor_generator.num_scales) * self.cls_out_channels,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                            padding='valid',
                            trainable=self.trainable,
                            name='rpn-class'
                        )
        self.conv_reg = tf.keras.layers.Conv2D(
                            len(self.anchor_generator.aspect_ratios * \
                                self.anchor_generator.num_scales) * 4,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                            padding='valid',
                            trainable=self.trainable,
                            name='rpn-box'
                        )
        
    def call(self, inputs, img_info, training=True, *args, **kwargs):
        scores_outputs = dict()
        box_outputs = dict()
        for level in range(self.anchor_generator.min_level, 
                           self.anchor_generator.max_level + 1):
            net = self.rpn_conv(inputs[level])
            scores_outputs[level] = self.conv_cls(net)
            box_outputs[level] = self.conv_reg(net)
        proposals = self.get_bboxes(scores_outputs,
                                    box_outputs,
                                    img_info,
                                    self.anchor_generator,
                                    training=training)
        return scores_outputs, box_outputs, proposals

@HEADS.register("StandardRPNHead")
def build_standard_rpn_head(cfg):
    head_type = StandardRPNHead
    return head_type(rpn_loss_cfg=dict(
                            min_level=cfg.MODEL.DENSE.MIN_LEVEL,
                            max_level=cfg.MODEL.DENSE.MAX_LEVEL,
                            box_loss_type=cfg.MODEL.DENSE.LOSS_TYPE,
                            train_batch_size_per_gpu=cfg.INPUT.TRAIN_BATCH_SIZE//MPI_size(),
                            rpn_batch_size_per_im=cfg.MODEL.DENSE.BATCH_SIZE_PER_IMAGE,
                            label_smoothing=cfg.MODEL.DENSE.LABEL_SMOOTHING,
                            rpn_box_loss_weight=cfg.MODEL.DENSE.LOSS_WEIGHT,
                         ),
                     anchor_generator_cfg=dict(
                            min_level=cfg.MODEL.DENSE.MIN_LEVEL, 
                            max_level=cfg.MODEL.DENSE.MAX_LEVEL,
                            num_scales=cfg.MODEL.DENSE.NUM_SCALES,
                            aspect_ratios=cfg.MODEL.DENSE.ASPECT_RATIOS,
                            anchor_scale=cfg.MODEL.DENSE.ANCHOR_SCALE,
                            image_size=cfg.INPUT.IMAGE_SIZE,
                         ),
                     roi_proposal_cfg = dict(
                             train_cfg = dict(
                                 rpn_pre_nms_topn=cfg.MODEL.DENSE.PRE_NMS_TOP_N_TRAIN,
                                 rpn_post_nms_topn=cfg.MODEL.DENSE.POST_NMS_TOP_N_TRAIN,
                                 rpn_nms_threshold=cfg.MODEL.DENSE.NMS_THRESH,
                             ),
                             test_cfg = dict(
                                 rpn_pre_nms_topn=cfg.MODEL.DENSE.PRE_NMS_TOP_N_TEST,
                                 rpn_post_nms_topn=cfg.MODEL.DENSE.POST_NMS_TOP_N_TEST,
                                 rpn_nms_threshold=cfg.MODEL.DENSE.NMS_THRESH,
                                 ),
                             rpn_min_size=cfg.MODEL.DENSE.MIN_SIZE,
                             use_custom_box_proposals_op=cfg.MODEL.DENSE.USE_FAST_BOX_PROPOSAL,
                             use_batched_nms=cfg.MODEL.DENSE.USE_BATCHED_NMS,
                             bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS,
                         ),
                     num_classes=1,
                     feat_channels=cfg.MODEL.DENSE.FEAT_CHANNELS,
                     trainable=cfg.MODEL.DENSE.TRAINABLE)