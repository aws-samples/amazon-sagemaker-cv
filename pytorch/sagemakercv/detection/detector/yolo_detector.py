import torch
import torch.nn as nn

# from ..backbone import build_backbone
# from ..dense_heads.yolo_head import build_yolo_head

class Yolov3Detector(nn.Module):
    def __init__(self, backbone, head, cfg):
        super(Yolov3Detector, self).__init__()
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.backbone = backbone
        self.head = head #yolo head
    
    def forward(self, images, targets = None):
        fpn_outputs = self.backbone(images.tensors)
        feature_maps = self.head(fpn_outputs)
        if targets:
            losses, parsed_targets = self.head.loss(images, feature_maps, targets)
            return losses, parsed_targets, feature_maps
        else:
            anchors, anchor_meta_data = self.head.anchor_generator(images, feature_maps)
            predictions = self.head.postprocessor(feature_maps, anchors)
            return predictions
    
def yolov3_model(backbone, model, cfg):
    return Yolov3Detector(backbone, model, cfg)