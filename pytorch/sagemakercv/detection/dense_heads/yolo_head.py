# Yolo head class that takes in fpn results and calculate prediction

import torch
from sagemakercv.layers.conv_module import ConvModule
from sagemakercv.detection import registry
from .yolo_loss import YoloLoss
from .yolo_anchor_generator import YOLOAnchorGenerator
from .inference import YoloPostProcessor
from sagemakercv.core.box_coder import BoxCoder
from sagemakercv.core.matcher import Matcher
from sagemakercv.core.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from sagemakercv.core.utils import cat


class YOLOV3Head(torch.nn.Module):
    
    def __init__(self,
                 num_classes,
                 anchor_generator,
                 proposal_matcher, 
                 fg_bg_sampler, 
                 bbox_coder,
                 in_channels=(512, 256, 128),
                 out_channels=(1024, 512, 256),
                 featmap_strides=[32, 16, 8]):
        super(YOLOV3Head, self).__init__()
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.anchor_generator = anchor_generator
        self.bbox_coder = bbox_coder
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.loss = YoloLoss(self.anchor_generator, 
                             self.proposal_matcher, 
                             self.fg_bg_sampler, 
                             self.bbox_coder)
        self.postprocessor = YoloPostProcessor(box_coder=self.bbox_coder)
#         self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.num_anchors = len([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]])
        self.convs_bridge = torch.nn.ModuleList()
        self.convs_pred = torch.nn.ModuleList()
        for i in range(self.num_levels):
            self.convs_bridge.append(ConvModule(self.in_channels[i],
                                                self.out_channels[i],
                                                3,
                                                padding=1))
            self.convs_pred.append(torch.nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1))
        assert len(self.anchor_generator.num_base_anchors) == len(featmap_strides)
    
    @property
    def num_levels(self):
        return len(self.featmap_strides)
    
    @property
    def num_attrib(self):
        return 5 + self.num_classes
    
    def forward(self, feat_maps):
        assert len(feat_maps) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feat_maps[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps)

# @registry.DENSE_HEADS.register("YOLOV3Head")
def build_yolo_head(cfg):
    anchors = YOLOAnchorGenerator(cfg.MODEL.YOLO.STRIDES, 
                                  cfg.MODEL.YOLO.BASE_SIZES)
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    fg_bg_sampler = BalancedPositiveNegativeSampler(
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        )
    box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)
    return YOLOV3Head(cfg.MODEL.YOLO.CLASSES,
                      anchors,
                      matcher,
                      fg_bg_sampler,
                      box_coder,
                      in_channels=(512, 256, 128),
                      out_channels=(1024, 512, 256),
                      featmap_strides=[32, 16, 8])
    
    
    
    
    