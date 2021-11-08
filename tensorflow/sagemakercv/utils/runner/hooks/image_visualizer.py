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
from concurrent.futures import ThreadPoolExecutor
from .hook import Hook
from sagemakercv.core import BoxDetector
from sagemakercv.data.coco.coco_labels import coco_categories
from sagemakercv.utils import visualization
from sagemakercv.utils.dist_utils import master_only
from ..builder import HOOKS

class ImageVisualizer(Hook):
    
    def __init__(self, 
                 interval=250, 
                 threshold=0.75):
        self.interval = interval
        self.threshold = threshold
    
    def build_image(self, 
                    image,
                    image_info,
                    boxes,
                    scores,
                    classes):
        image = visualization.restore_image(image, image_info)
        detection_image = visualization.build_image(image, 
                                                    boxes, 
                                                    scores, 
                                                    classes, 
                                                    class_names=coco_categories, 
                                                    threshold=self.threshold)
        detection_image = tf.expand_dims(detection_image, axis=0)
        return detection_image
    
    def build_image_batch(self, runner):
        data_batch = runner.data_batch
        outputs = runner.outputs
        detections = self.inference_detector(outputs['class_outputs'],
                                             outputs['box_outputs'],
                                             outputs['box_rois'],
                                             outputs['image_info'])
        
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']
        images = data_batch[0]['images']
        image_info = data_batch[0]['image_info']
        batch_size = boxes.shape[0]
        detection_images = [self.build_image(images[i], image_info[i], boxes[i], scores[i], classes[i]) \
                            for i in range(batch_size)]
        return detection_images
    
    def images_to_tensorboard(self, runner):
        detection_images = self.build_image_batch(runner)
        writer = tf.summary.create_file_writer(runner.tensorboard_dir)
        with writer.as_default():
            for i, image in enumerate(detection_images):
                tf.summary.image(f"image_{i}",
                                 image,
                                 step=runner.iter)
        writer.close()
        
    @master_only
    def before_run(self, runner):
        self.inference_detector = BoxDetector(use_batched_nms=runner.cfg.MODEL.INFERENCE.USE_BATCHED_NMS,
                                              rpn_post_nms_topn=runner.cfg.MODEL.INFERENCE.POST_NMS_TOPN,
                                              detections_per_image=runner.cfg.MODEL.INFERENCE.DETECTIONS_PER_IMAGE,
                                              test_nms=runner.cfg.MODEL.INFERENCE.DETECTOR_NMS,
                                              class_agnostic_box=runner.cfg.MODEL.INFERENCE.CLASS_AGNOSTIC,
                                              bbox_reg_weights=runner.cfg.MODEL.BBOX_REG_WEIGHTS)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    @master_only
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval):
            self.thread_pool.submit(self.images_to_tensorboard, runner)

@HOOKS.register("ImageVisualizer")
def build_image_visualizer(cfg):
    return ImageVisualizer(interval=cfg.MODEL.INFERENCE.VISUALIZE_INTERVAL, 
                           threshold=cfg.MODEL.INFERENCE.VISUALIZE_THRESHOLD)
