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

from sagemakercv.utils import Registry

ANCHORS = Registry()
ROI_SELECTORS = Registry()
INFERENCE_DETECTORS = Registry()
ENCODERS = Registry()
ROI_EXTRACTORS = Registry()
SAMPLERS = Registry()

def build_anchor_generator(cfg):
    '''
    Reads standard configuration file to build
    anchor generator
    '''
    anchor_type = ANCHORS['AnchorGenerator']
    return anchor_type(min_level=cfg.MODEL.RPN.MIN_LEVEL,
                       max_level=cfg.MODEL.RPN.MAX_LEVEL,
                       num_scales=cfg.MODEL.RPN.NUM_SCALES,
                       aspect_ratios=cfg.MODEL.RPN.ASPECT_RATIOS,
                       anchor_scale=cfg.MODEL.RPN.ANCHOR_SCALE,
                       image_size=cfg.INPUT.IMAGE_SIZE)

def build_anchor_labeler(cfg, anchor_generator):
    '''
    Reads standard configuration file to build
    anchor labeler
    '''
    anchor_type = ANCHORS['AnchorLabeler']
    return anchor_type(anchors=anchor_generator,
                       num_classes=cfg.INPUT.NUM_CLASSES,
                       match_threshold=cfg.MODEL.RPN.POSITIVE_OVERLAP,
                       unmatched_threshold=cfg.MODEL.RPN.NEGATIVE_OVERLAP,
                       rpn_batch_size_per_im=cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
                       rpn_fg_fraction=cfg.MODEL.RPN.FG_FRACTION)

def build_anchors(cfg):
    anchor_generator = build_anchor_generator(cfg)
    anchor_labeler = build_anchor_labeler(cfg, anchor_generator)
    return anchor_generator, anchor_labeler

def build_roi_selector(cfg):
    '''
    Reads standard configuration file to build
    NMS region of interest selector
    '''
    roi_selector = ROI_SELECTORS["ProposeROIs"]
    return roi_selector(train_cfg = dict(
                             rpn_pre_nms_topn=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN,
                             rpn_post_nms_topn=cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN,
                             rpn_nms_threshold=cfg.MODEL.RPN.NMS_THRESH,
                             ),
                         test_cfg = dict(
                             rpn_pre_nms_topn=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST,
                             rpn_post_nms_topn=cfg.MODEL.RPN.POST_NMS_TOP_N_TEST,
                             rpn_nms_threshold=cfg.MODEL.RPN.NMS_THRESH,
                             ),
                         rpn_min_size=cfg.MODEL.RPN.MIN_SIZE,
                         use_custom_box_proposals_op=cfg.MODEL.RPN.USE_FAST_BOX_PROPOSAL,
                         use_batched_nms=cfg.MODEL.RPN.USE_BATCHED_NMS,
                         bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS
                        )
