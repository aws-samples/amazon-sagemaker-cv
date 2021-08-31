# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# revised to suit yolov3 prediction style

import torch
import torch.nn.functional as F
from torch import nn

from sagemakercv.core.structures.bounding_box import BoxList
from sagemakercv.core.structures.boxlist_ops import boxlist_nms
from sagemakercv.core.structures.boxlist_ops import cat_boxlist
from sagemakercv.core.box_coder import BoxCoder
from sagemakercv.detection.roi_heads.box_head.inference import PostProcessor
from sagemakercv import _C as C

class YoloPostProcessor(PostProcessor):
    
    def __init__(
            self,
            score_thresh=0.05,
            nms=0.5,
            detections_per_img=100,
            box_coder=None,
        ):
        super(YoloPostProcessor, self).__init__(score_thresh,
                                                nms,
                                                detections_per_img,
                                                box_coder)
    
    def forward(self, feature_maps, boxes):
        features_cat = torch.cat([i.permute(0, 2, 3, 1).reshape(feature_maps[0].shape[0], -1, 85) \
                                  for i in feature_maps], dim=1).reshape(-1, 85)
        features_dict = {}
        box_regression = features_cat[:,:4]
        features_dict['obj'] = torch.sigmoid(features_cat[:, 4])
        class_prob = torch.softmax(features_cat[:, 5:], -1) * \
                                 torch.stack([features_dict['obj'] for _ in range(80)], dim=1)
            
        num_classes = class_prob.shape[1]
        
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        
        proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
        
        proposals = proposals.repeat(1, num_classes)
        
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        
        results = []

        for prob, boxes_per_img, image_shape in zip(
                    class_prob, proposals, image_shapes
                ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results
    
    def filter_results(self, boxlist, num_classes):
        boxes = boxlist.bbox.reshape(-1, num_classes * 4).float()
        scores = boxlist.get_field("scores").reshape(-1, num_classes).float()
        image_shape = boxlist.size
        device = scores.device
        result = []
        boxes = boxes.view(-1, num_classes, 4).permute(1,0,2).contiguous() # [1:,:,:]
        scores = scores.permute(1,0).contiguous() # [1:,:]
        inds_all = scores > 0.05
        num_detections = scores.size(1)
        num_boxes_per_class = [num_detections] * num_classes
        num_boxes_tensor = torch.tensor(num_boxes_per_class, device=boxes.device, dtype=torch.int32)
        sorted_idx = scores.argsort(dim=1, descending=True)  
        batch_idx = torch.arange(num_classes, device=device)[:, None]
        boxes = boxes[batch_idx, sorted_idx]
        scores = scores[batch_idx, sorted_idx]
        inds_all = inds_all[batch_idx, sorted_idx]
        boxes = boxes.reshape(-1, 4)
        keep_inds_batched = C.nms_batched(boxes, num_boxes_per_class, num_boxes_tensor, inds_all.byte(), 0.5)
        keep_inds = keep_inds_batched.view(-1).nonzero().squeeze(1)
        boxes = boxes.reshape(-1,4).index_select(dim=0, index=keep_inds)
        scores = scores.reshape(-1).index_select(dim=0, index=keep_inds)
        labels_all = torch.tensor(num_detections * list(range(1, num_classes+1)), device=device, dtype=torch.int64)
        labels_all = labels_all.view(num_detections, (num_classes)).permute(1,0).contiguous().reshape(-1)
        labels = labels_all.index_select(dim=0, index=keep_inds)
        result = BoxList(boxes, image_shape, mode="xyxy")
        result.add_field("scores",scores)
        result.add_field("labels", labels)
        number_of_detections = len(result)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result
        