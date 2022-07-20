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
from sagemakercv.core import AnchorGenerator, ProposeROIs

class AnchorHead(tf.keras.Model):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605
    def __init__(self,
                 anchor_generator_cfg = dict(
                     min_level = 2, 
                     max_level = 6, 
                     num_scales = 1, 
                     aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], 
                     anchor_scale = 8.0, 
                     image_size = (832, 1344),
                 ),
                 roi_proposal_cfg = dict(
                     train_cfg = dict(
                         rpn_pre_nms_topn=2000,
                         rpn_post_nms_topn=1000,
                         rpn_nms_threshold=0.7,
                     ),
                     test_cfg = dict(
                         rpn_pre_nms_topn=1000,
                         rpn_post_nms_topn=1000,
                         rpn_nms_threshold=0.7,
                         ),
                     rpn_min_size=0.,
                     use_custom_box_proposals_op=True,
                     use_batched_nms=False,
                     bbox_reg_weights=None,
                 ),
                 num_classes=1,
                 feat_channels=256,
                 trainable=True,
                 ):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes
        self.anchor_generator = AnchorGenerator(**anchor_generator_cfg)
        self.roi_proposal = ProposeROIs(**roi_proposal_cfg)
        self.trainable = trainable
        self._init_layers()
        
    def _init_layers(self):
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
    
    def call(self, inputs, img_info, gt_boxes=None, gt_labels=None, training=True, *args, **kwargs):
        cls_scores = self.conv_cls(inputs)
        bbox_preds = self.conv_reg(inputs)
        proposals = self.get_bboxes(cls_scores,
                                    bbox_preds,
                                    img_info,
                                    self.anchor_generator,
                                    gt_boxes=gt_boxes,
                                    gt_labels=gt_labels,
                                    training=training)
        return cls_scores, bbox_preds, proposals
    
    def get_bboxes(self, 
                   cls_scores,
                   bbox_preds,
                   img_info,
                   anchors,
                   training=True):
        rpn_box_rois, rpn_box_scores = self.roi_proposal(cls_scores,
                                                           bbox_preds,
                                                           img_info,
                                                           anchors,
                                                           training=training)
        return rpn_box_rois, rpn_box_scores