#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, Facebook, Inc. All rights reserved.
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

"""Centralized catalog of paths."""

import os

class DatasetCatalog(object):
    DATA_DIR = os.path.expanduser("~/data")
    DATASETS = {
        "coco_2017_train": {
            "img_file_pattern": "coco/train/train*",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_file_pattern": "coco/val/val*",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
    }
    
    WEIGHTS = os.path.join(DATA_DIR, 'weights/tf/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603')
    
    OUTPUT_DIR = "/home/ubuntu/models"
    