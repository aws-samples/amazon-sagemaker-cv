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

from .anchors import AnchorGenerator, AnchorLabeler
from .roi_ops import ProposeROIs
from .training_ops import TargetEncoder, RandomSampler
from .postprocess_ops import BoxDetector
from .spatial_transform_ops import GenericRoIExtractor

from .builder import (ANCHORS,
                      ENCODERS,
                      ROI_EXTRACTORS,
                      INFERENCE_DETECTORS,
                      build_anchor_generator, 
                      build_anchor_labeler, 
                      build_anchors, 
                      build_roi_selector)