# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from sagemakercv.core.structures.image_list import to_image_list
from sagemakercv.layers.nhwc import nchw_to_nhwc_transform, nhwc_to_nchw_transform

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn, build_rpn_head
from ..roi_heads.roi_heads import build_roi_heads

class Graphable(nn.Module):
    def __init__(self, cfg):
        super(Graphable, self).__init__()
        self.backbone = build_backbone(cfg)
        self.anchor_generator, self.head = build_rpn_head(cfg)
        self.nhwc = cfg.OPT_LEVEL=="O4"
    
    def forward(self, images_tensor, image_sizes_tensor):
        features = self.backbone(images_tensor)
        if self.nhwc:
            features = [tuple(features[0:5]),
                        tuple(features[5:10])]
            objectness, rpn_box_regression = self.head(features[1])
            with torch.no_grad():
                anchor_boxes, anchor_visibility = self.anchor_generator(image_sizes_tensor.int(), features[1])
            return features[0] + tuple(objectness) + tuple(rpn_box_regression) + (anchor_boxes, anchor_visibility)
        else:
            objectness, rpn_box_regression = self.head(features)
            with torch.no_grad():
                anchor_boxes, anchor_visibility = self.anchor_generator(image_sizes_tensor.int(), features)
            return features + tuple(objectness) + tuple(rpn_box_regression) + (anchor_boxes, anchor_visibility)


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        
        self.cfg = cfg.clone()
        self.graphable = Graphable(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.nhwc = cfg.OPT_LEVEL=="O4"

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        if self.nhwc:
            images.tensors = nchw_to_nhwc_transform(images.tensors)
        flat_res = self.graphable(images.tensors, images.image_sizes_tensor)
        features, objectness, rpn_box_regression, anchor_boxes, anchor_visibility = \
        flat_res[0:5], list(flat_res[5:10]), list(flat_res[10:15]), flat_res[15], flat_res[16]
        #features = self.backbone(images.tensors)
        #proposals, proposal_losses = self.rpn(images, features, targets)
        #proposals, proposal_losses = self.rpn(images, features, targets) if not self.nhwc else \
        #                        self.rpn(images, features[1], targets)
        proposals, proposal_losses = self.rpn(images, anchor_boxes, anchor_visibility, 
                                              objectness, rpn_box_regression, targets)

        if self.roi_heads:
            #x, result, detector_losses = self.roi_heads(features, proposals, targets) if not self.nhwc \
            #        else self.roi_heads(features[0], proposals, targets)
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        ## for NHWC layout case, features[0] are NHWC features, and [1] NCHW
        else:
            # RPN-only models don't have roi_heads
            ## TODO: take care of NHWC/NCHW cases for RPN-only case 
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
