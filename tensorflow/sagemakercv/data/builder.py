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

import os
from sagemakercv.utils import Registry
from sagemakercv.utils.dist_utils import MPI_size

DATASETS = Registry()

def build_dataset(cfg, mode='train'):
    if mode=='train':
        file_pattern = cfg.PATHS.TRAIN_FILE_PATTERN
    elif mode in ['eval', 'infer']:
        file_pattern = cfg.PATHS.VAL_FILE_PATTERN
    else:
        raise NotImplementedError
    params=dict(
                visualize_images_summary=cfg.INPUT.VISUALIZE_IMAGES_SUMMARY,
                image_size=cfg.INPUT.IMAGE_SIZE,
                min_level=cfg.MODEL.DENSE.MIN_LEVEL,
                max_level=cfg.MODEL.DENSE.MAX_LEVEL,
                num_scales=cfg.MODEL.DENSE.NUM_SCALES,
                aspect_ratios=cfg.MODEL.DENSE.ASPECT_RATIOS,
                anchor_scale=cfg.MODEL.DENSE.ANCHOR_SCALE,
                include_mask=cfg.MODEL.INCLUDE_MASK,
                skip_crowd_during_training=cfg.INPUT.SKIP_CROWDS_DURING_TRAINING,
                include_groundtruth_in_features=cfg.INPUT.INCLUDE_GROUNDTRUTH_IN_FEATURES,
                use_category=cfg.INPUT.USE_CATEGORY,
                augment_input_data=cfg.INPUT.AUGMENT_INPUT_DATA,
                gt_mask_size=cfg.INPUT.GT_MASK_SIZE,
                num_classes=cfg.INPUT.NUM_CLASSES,
                rpn_positive_overlap=cfg.MODEL.DENSE.POSITIVE_OVERLAP,
                rpn_negative_overlap=cfg.MODEL.DENSE.NEGATIVE_OVERLAP,
                rpn_batch_size_per_im=cfg.MODEL.DENSE.BATCH_SIZE_PER_IMAGE,
                rpn_fg_fraction=cfg.MODEL.DENSE.FG_FRACTION,
            )
    dataset = DATASETS[cfg.INPUT.DATALOADER]
    return dataset(file_pattern,
                   params,
                   mode=mode,
                   batch_size=cfg.INPUT.TRAIN_BATCH_SIZE//MPI_size() if mode=='train' \
                              else cfg.INPUT.EVAL_BATCH_SIZE//MPI_size(),
                   use_instance_mask=cfg.MODEL.INCLUDE_MASK if mode=='train' else False,
                   )()