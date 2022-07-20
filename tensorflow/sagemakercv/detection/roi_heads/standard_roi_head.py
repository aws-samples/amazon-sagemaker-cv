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
from sagemakercv.core import training_ops, GenericRoIExtractor, TargetEncoder, BoxDetector, RandomSampler
from ..builder import HEADS, build_box_head, build_mask_head

class StandardRoIHead(tf.keras.Model):
    """Simplest base roi head including one bbox head and one mask head."""
    
    def __init__(self,
                 bbox_head,
                 bbox_roi_extractor,
                 bbox_sampler,
                 box_encoder,
                 inference_detector,
                 mask_head=None,
                 mask_roi_extractor=None,
                 name="StandardRoIHead",
                 trainable=True,
                 *args,
                 **kwargs,
                 ):
        super(StandardRoIHead, self).__init__(name=name, trainable=trainable, *args, **kwargs)
        self.bbox_head = bbox_head
        self.bbox_roi_extractor = bbox_roi_extractor
        self.bbox_sampler = bbox_sampler
        self.box_encoder = box_encoder
        self.mask_head = mask_head
        self.mask_roi_extractor = mask_roi_extractor if mask_roi_extractor is not None \
                                  else self.bbox_roi_extractor
        self.inference_detector = inference_detector
    
    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None
    
    def call(self,
             fpn_feats,
             img_info,
             proposals,
             gt_bboxes=None,
             gt_labels=None,
             gt_masks=None,
             training=True):
        model_outputs=dict()
        
        if training:
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = self.bbox_sampler(proposals,
                                                                                                gt_bboxes, 
                                                                                                gt_labels)
        else:
            rpn_box_rois = proposals
            
        box_roi_features = self.bbox_roi_extractor(fpn_feats, rpn_box_rois)
        
        class_outputs, box_outputs, _ = self.bbox_head(inputs=box_roi_features)
        
        if not training:
            model_outputs.update(self.inference_detector(class_outputs, box_outputs, rpn_box_rois, img_info))
            model_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
                                  'box_outputs': box_outputs,
                                  'anchor_boxes': rpn_box_rois})
        else:
            if self.bbox_head.loss.box_loss_type not in ["giou", "ciou"]:
                encoded_box_targets = self.box_encoder(boxes=rpn_box_rois,
                                                       gt_bboxes=box_targets,
                                                       gt_labels=class_targets)
            model_outputs.update({
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets if self.bbox_head.loss.box_loss_type \
                                                      not in ["giou", "ciou"] \
                                                      else box_targets,
                'box_rois': rpn_box_rois,
            })
            total_loss, class_loss, box_loss = self.bbox_head.loss(model_outputs['class_outputs'],
                                                                   model_outputs['box_outputs'],
                                                                   model_outputs['class_targets'],
                                                                   model_outputs['box_targets'],
                                                                   model_outputs['box_rois'],
                                                                   img_info)
            model_outputs.update({
                'total_loss_bbox': total_loss,
                'class_loss': class_loss,
                'box_loss': box_loss
            })
        if not self.with_mask:
            return model_outputs
        if not training:
            return self.call_mask(model_outputs, fpn_feats, training=False)
        max_fg = int(self.bbox_sampler.batch_size_per_im * self.bbox_sampler.fg_fraction)
        return self.call_mask(model_outputs,
                              fpn_feats,
                              class_targets=class_targets,
                              box_targets=box_targets,
                              rpn_box_rois=rpn_box_rois,
                              proposal_to_label_map=proposal_to_label_map,
                              gt_masks=gt_masks,
                              max_fg=max_fg,
                              training=True)
    
    def call_mask(self, 
                  model_outputs,
                  fpn_feats,
                  class_targets=None,
                  box_targets=None,
                  rpn_box_rois=None,
                  proposal_to_label_map=None,
                  gt_masks=None,
                  max_fg=None,
                  training=True):
        if not training:
            selected_box_rois = model_outputs['detection_boxes']
            class_indices = model_outputs['detection_classes']
            class_indices = tf.cast(class_indices, dtype=tf.int32)
            
        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=max_fg
            )
            
            class_indices = selected_class_targets
            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)
            
        mask_roi_features = self.mask_roi_extractor(
                fpn_feats,
                selected_box_rois,
            )
        
        mask_outputs = self.mask_head(inputs=mask_roi_features, class_indices=class_indices)
        
        if training:
            mask_targets = training_ops.get_mask_targets(
                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=gt_masks,
                output_size=self.mask_head._mrcnn_resolution
            )

            model_outputs.update({
                'mask_outputs': mask_outputs,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })
            
            mask_loss = self.mask_head.loss(model_outputs['mask_outputs'],
                                             model_outputs['mask_targets'],
                                             model_outputs['selected_class_targets'],)
            model_outputs.update({'mask_loss': mask_loss})

        else:
            model_outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_outputs),
            })

        return model_outputs

@HEADS.register("StandardRoIHead")
def build_standard_roi_head(cfg):
    roi_head = StandardRoIHead
    bbox_head = build_box_head(cfg)
    bbox_roi_extractor = GenericRoIExtractor(cfg.MODEL.FRCNN.ROI_SIZE,
                                            cfg.MODEL.FRCNN.GPU_INFERENCE)
    bbox_sampler = RandomSampler(batch_size_per_im=cfg.MODEL.RCNN.BATCH_SIZE_PER_IMAGE,
                                 fg_fraction=cfg.MODEL.RCNN.FG_FRACTION, 
                                 fg_thresh=cfg.MODEL.RCNN.THRESH,     
                                 bg_thresh_hi=cfg.MODEL.RCNN.THRESH_HI, 
                                 bg_thresh_lo=cfg.MODEL.RCNN.THRESH_LO)
    box_encoder = TargetEncoder(bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS)
    inference_detector = BoxDetector(use_batched_nms=cfg.MODEL.INFERENCE.USE_BATCHED_NMS,
                                     rpn_post_nms_topn=cfg.MODEL.INFERENCE.POST_NMS_TOPN,
                                     detections_per_image=cfg.MODEL.INFERENCE.DETECTIONS_PER_IMAGE,
                                     test_nms=cfg.MODEL.INFERENCE.DETECTOR_NMS,
                                     class_agnostic_box=cfg.MODEL.INFERENCE.CLASS_AGNOSTIC,
                                     bbox_reg_weights=cfg.MODEL.BBOX_REG_WEIGHTS)

    if cfg.MODEL.INCLUDE_MASK:
        mask_roi_extractor = GenericRoIExtractor(cfg.MODEL.MRCNN.ROI_SIZE,
                                                 cfg.MODEL.MRCNN.GPU_INFERENCE)
        mask_head = build_mask_head(cfg)
    else:
        mask_head = None
        mask_roi_extractor = None
    return roi_head(bbox_head=bbox_head,
                    bbox_roi_extractor=bbox_roi_extractor,
                    bbox_sampler=bbox_sampler,
                    box_encoder=box_encoder,
                    inference_detector=inference_detector,
                    mask_head=mask_head,
                    mask_roi_extractor=mask_roi_extractor)
