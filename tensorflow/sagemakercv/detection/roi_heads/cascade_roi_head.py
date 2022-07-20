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
from sagemakercv.core import (training_ops, 
                              GenericRoIExtractor, 
                              TargetEncoder, 
                              BoxDetector, 
                              RandomSampler,
                              box_utils)
from ..builder import HEADS, build_box_head, build_mask_head
from .bbox_heads import StandardBBoxHead
from .standard_roi_head import StandardRoIHead

class CascadeRoIHead(StandardRoIHead):
    def __init__(self,
                 bbox_head,
                 bbox_roi_extractor,
                 bbox_sampler,
                 box_encoder,
                 inference_detector,
                 stage_weights,
                 mask_head=None,
                 mask_roi_extractor=None,
                 name="CascadeRoIHead",
                 trainable=True,
                 *args,
                 **kwargs,
                 ):
        super(CascadeRoIHead, self).__init__(bbox_head=bbox_head,
                                             bbox_roi_extractor=bbox_roi_extractor,
                                             bbox_sampler=bbox_sampler,
                                             box_encoder=box_encoder,
                                             inference_detector=inference_detector,
                                             mask_head=mask_head,
                                             mask_roi_extractor=mask_roi_extractor,
                                             name=name, 
                                             trainable=trainable, 
                                             *args, 
                                             **kwargs)
        self.stage_weights = [i/sum(stage_weights) for i in stage_weights]
        self.num_stages = len(self.bbox_head)
    
    @tf.custom_gradient
    def scale_gradient(self, x):
        return x, lambda dy: dy * (1.0 / self.num_stages)
    
    def call(self,
             fpn_feats,
             img_info,
             proposals,
             gt_bboxes=None,
             gt_labels=None,
             gt_masks=None,
             training=True):
        model_outputs=dict()
        for stage in range(self.num_stages):
            if training:
                box_targets, class_targets, rpn_box_rois, proposal_to_label_map = self.bbox_sampler[stage](proposals,
                                                                                                           gt_bboxes, 
                                                                                                           gt_labels)
            else:
                rpn_box_rois = proposals
            box_roi_features = self.bbox_roi_extractor(fpn_feats, rpn_box_rois)
            if training:
                box_roi_features = self.scale_gradient(box_roi_features)
            class_outputs, box_outputs, _ = self.bbox_head[stage](inputs=box_roi_features)
            if training:
                if self.bbox_head[stage].loss.box_loss_type not in ["giou", "ciou"]:
                    encoded_box_targets = self.box_encoder[stage](boxes=rpn_box_rois,
                                                                  gt_bboxes=box_targets,
                                                                  gt_labels=class_targets)
                model_outputs.update({
                    f'class_outputs_{stage}': class_outputs,
                    f'box_outputs_{stage}': box_outputs,
                    f'class_targets_{stage}': class_targets,
                    f'box_targets_{stage}': encoded_box_targets,
                    f'box_rois_{stage}': rpn_box_rois,
                })
                
                total_loss, class_loss, box_loss = \
                self.bbox_head[stage].loss(model_outputs[f'class_outputs_{stage}'],
                                            model_outputs[f'box_outputs_{stage}'],
                                            model_outputs[f'class_targets_{stage}'],
                                            model_outputs[f'box_targets_{stage}'],
                                            model_outputs[f'box_rois_{stage}'],
                                            img_info)
                
                model_outputs.update({
                    f'total_loss_{stage}': total_loss, #* self.stage_weights[stage],
                    f'class_loss_{stage}': class_loss, #* self.stage_weights[stage],
                    f'box_loss_{stage}': box_loss, #* self.stage_weights[stage],
                })
            
            if stage < self.num_stages - 1:
                new_proposals = box_utils.decode_boxes(box_outputs[:,:,4:], 
                                                       rpn_box_rois, 
                                                       weights=self.bbox_head[stage].loss.bbox_reg_weights)
                height = tf.expand_dims(img_info[:, 0:1], axis=-1)
                width = tf.expand_dims(img_info[:, 1:2], axis=-1)
                boxes = box_utils.clip_boxes(new_proposals, (height, width))
                proposals = tf.stop_gradient(new_proposals)
        
        if not training:
            model_outputs.update(self.inference_detector(class_outputs, box_outputs, rpn_box_rois, img_info))
            model_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
                                  'box_outputs': box_outputs,
                                  'anchor_boxes': rpn_box_rois})
        else:
            # add final outputs for visualizer
            model_outputs.update({
                    'class_outputs': class_outputs,
                    'box_outputs': box_outputs,
                    'class_targets': class_targets,
                    'box_targets': encoded_box_targets,
                    'box_rois': rpn_box_rois,
                })
                
        if not self.with_mask:
            return model_outputs
        
        if not training:
            return self.call_mask(model_outputs, fpn_feats, training=False)
        max_fg = int(self.bbox_sampler[-1].batch_size_per_im * self.bbox_sampler[-1].fg_fraction)
        print(class_targets)
        return self.call_mask(model_outputs,
                              fpn_feats,
                              class_targets=class_targets,
                              box_targets=box_targets,
                              rpn_box_rois=rpn_box_rois,
                              proposal_to_label_map=proposal_to_label_map,
                              gt_masks=gt_masks,
                              max_fg=max_fg,
                              training=True)

@HEADS.register("CascadeRoIHead")
def build_cascade_roi_head(cfg):
    roi_head = CascadeRoIHead
    assert len(cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS) == \
           len(cfg.MODEL.RCNN.CASCADE.THRESHOLDS) == \
           len(cfg.MODEL.RCNN.CASCADE.STAGE_WEIGHTS)
    num_stages = len(cfg.MODEL.RCNN.CASCADE.THRESHOLDS)
    bbox_heads = build_box_head(cfg)
    bbox_roi_extractor = GenericRoIExtractor(cfg.MODEL.FRCNN.ROI_SIZE,
                                            cfg.MODEL.FRCNN.GPU_INFERENCE)
    bbox_samplers = [RandomSampler(batch_size_per_im=cfg.MODEL.RCNN.BATCH_SIZE_PER_IMAGE,
                                 fg_fraction=cfg.MODEL.RCNN.FG_FRACTION, 
                                 fg_thresh=stage_tresh,     
                                 bg_thresh_hi=stage_tresh, 
                                 bg_thresh_lo=cfg.MODEL.RCNN.THRESH_LO) \
                                 for stage_tresh in cfg.MODEL.RCNN.CASCADE.THRESHOLDS]
    box_encoders = [TargetEncoder(bbox_reg_weights=bbox_weights) \
                    for bbox_weights in cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS]
    inference_detector = BoxDetector(use_batched_nms=cfg.MODEL.INFERENCE.USE_BATCHED_NMS,
                                     rpn_post_nms_topn=cfg.MODEL.INFERENCE.POST_NMS_TOPN,
                                     detections_per_image=cfg.MODEL.INFERENCE.DETECTIONS_PER_IMAGE,
                                     test_nms=cfg.MODEL.INFERENCE.DETECTOR_NMS,
                                     class_agnostic_box=cfg.MODEL.FRCNN.CLASS_AGNOSTIC,
                                     bbox_reg_weights=cfg.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS[-1])
    if cfg.MODEL.INCLUDE_MASK:
        mask_roi_extractor = GenericRoIExtractor(cfg.MODEL.MRCNN.ROI_SIZE,
                                                 cfg.MODEL.MRCNN.GPU_INFERENCE)
        mask_head = build_mask_head(cfg)
    else:
        mask_head = None
        mask_roi_extractor = None
    return roi_head(bbox_head=bbox_heads,
                    bbox_roi_extractor=bbox_roi_extractor,
                    bbox_sampler=bbox_samplers,
                    box_encoder=box_encoders,
                    inference_detector=inference_detector,
                    stage_weights=cfg.MODEL.RCNN.CASCADE.STAGE_WEIGHTS,
                    mask_head=mask_head,
                    mask_roi_extractor=mask_roi_extractor)
