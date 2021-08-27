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

BACKBONES = Registry()
NECKS = Registry()
ROI_EXTRACTORS = Registry()
SHARED_HEADS = Registry()
HEADS = Registry()
DETECTORS = Registry()

def build_backbone(cfg):
    backbone_type = BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY]
    return backbone_type(sub_type=cfg.MODEL.BACKBONE.CONV_BODY,
                         data_format=cfg.MODEL.BACKBONE.DATA_FORMAT,
                         trainable=cfg.MODEL.BACKBONE.TRAINABLE,
                         finetune_bn=cfg.MODEL.BACKBONE.FINETUNE_BN,
                         norm_type=cfg.MODEL.BACKBONE.NORM_TYPE)

def build_dense_head(cfg):
    return HEADS[cfg.MODEL.DENSE.RPN_HEAD](cfg)

def build_neck(cfg):
    return NECKS[cfg.MODEL.BACKBONE.NECK](cfg)

def build_box_head(cfg):
    return HEADS[cfg.MODEL.FRCNN.BBOX_HEAD](cfg)

def build_mask_head(cfg):
    return HEADS[cfg.MODEL.MRCNN.MASK_HEAD](cfg)

def build_roi_head(cfg):
    return HEADS[cfg.MODEL.RCNN.ROI_HEAD](cfg)

def build_detector(cfg):
    return DETECTORS[cfg.MODEL.DETECTOR](cfg)
