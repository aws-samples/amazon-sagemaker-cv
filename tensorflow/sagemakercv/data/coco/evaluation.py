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

import numpy as np
import cv2
import pycocotools.mask as maskUtils
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def process_boxes(image_info, box_coordinates):
    """Process the model prediction for COCO eval."""
    '''image_info = prediction['image_info']
    box_coordinates = prediction['detection_boxes']'''
    processed_box_coordinates = np.zeros_like(box_coordinates)

    for image_id in range(box_coordinates.shape[0]):
        scale = image_info[image_id][2]

        for box_id in range(box_coordinates.shape[1]):
            # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
            # Map [y1, x1, y2, x2] -> [x1, y1, w, h] and multiply detections
            # by image scale.
            y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
            new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
            processed_box_coordinates[image_id, box_id, :] = new_box

    #prediction['detection_boxes'] = processed_box_coordinates
    return processed_box_coordinates

def generate_segmentation_from_masks(masks,
                                     detected_boxes,
                                     image_height,
                                     image_width,
                                     is_image_mask=False):
    """Generates segmentation result from instance masks.

    Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.
    is_image_mask: bool. True: input masks are whole-image masks. False: input
      masks are bounding-box level masks.

    Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
    """

    def expand_boxes(boxes, scale):
        """Expands an array of boxes by a given scale."""
        # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227
        # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
        # whereas `boxes` here is in [x1, y1, w, h] form
        w_half = boxes[:, 2] * .5
        h_half = boxes[:, 3] * .5
        x_c = boxes[:, 0] + w_half
        y_c = boxes[:, 1] + h_half

        w_half *= scale
        h_half *= scale

        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half

        return boxes_exp

    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.

    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width,
              (mask_height + 2.0) / mask_height)
    ref_boxes = expand_boxes(detected_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []
    for mask_ind, mask in enumerate(masks):
        im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        if is_image_mask:
            # Process whole-image masks.
            im_mask[:, :] = mask[:, :]
        else:
            # Process mask inside bounding boxes.
            padded_mask[1:-1, 1:-1] = mask[:, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)
            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)
            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, image_width)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, image_height)
            im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (
                  x_0 - ref_box[0]):(x_1 - ref_box[0])]
        segms.append(im_mask)

    segms = np.array(segms)
    assert masks.shape[0] == segms.shape[0]
    return segms

def format_predictions(image_id,
                       detection_boxes,
                       detection_scores,
                       detection_classes,
                       rles):
    box_predictions = []
    mask_predictions = []
    
    detection_count = len(detection_scores)
    for i in range(detection_count):
        box_predictions.append({'image_id': int(image_id),
                                'category_id': int(detection_classes[i]),
                                'bbox': list(map(lambda x: float(round(x, 2)), detection_boxes[i])),
                                'score': float(detection_scores[i])})
        if rles:
            segmentation = {'size': rles[i]['size'],
                            'counts': rles[i]['counts'].decode()} 
            mask_predictions.append({'image_id': int(image_id),
                                     'category_id': int(detection_classes[i]),
                                     'score': float(detection_scores[i]),
                                     'segmentation': segmentation})
    return box_predictions, mask_predictions


def process_prediction(prediction):
    prediction.update({'detection_boxes':
                       process_boxes(prediction['image_info'], 
                                     prediction['detection_boxes'])})
    batch_size = prediction['num_detections'].shape[0]
    
    box_predictions = []
    mask_predictions = []
    imgIds = []
    
    for i in range(batch_size):
        detection_boxes = prediction['detection_boxes'][i]
        detection_classes = prediction['detection_classes'][i]
        detection_scores = prediction['detection_scores'][i]
        source_id = prediction['source_ids'][i]
        detection_masks = prediction['detection_masks'][i] if 'detection_masks' in prediction.keys() else None
        segments = generate_segmentation_from_masks(detection_masks,
                                     detection_boxes,
                                     int(prediction['image_info'][i][3]),
                                     int(prediction['image_info'][i][4])) if detection_masks is not None else None
        rles = None
        if detection_masks is not None:
            rles = [
                maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
                for instance_mask in segments
            ]
        formatted_predictions = format_predictions(source_id,
                                               detection_boxes,
                                               detection_scores,
                                               detection_classes,
                                               rles)
        imgIds.append(source_id)
        box_predictions.extend(formatted_predictions[0])
        mask_predictions.extend(formatted_predictions[1])
    
    
    return imgIds, box_predictions, mask_predictions

def build_output_dict(iou, stats, verbose=False):
    if verbose:
        format_string = ["AP 0.50:0.95 all",
                         "AP 0.50 all",
                         "AP 0.75 all",
                         "AP 0.50:0.95 small",
                         "AP 0.50:0.95 medium",
                         "AP 0.50:0.95 large",
                         "AR 0.50:0.95 all",
                         "AR 0.50 all",
                         "AR 0.75 all",
                         "AR 0.50:0.95 small",
                         "AR 0.50:0.95 medium",
                         "AR 0.50:0.95 large"]
        stat_dict = {"{0} {1}".format(iou, i): j for i,j in zip(format_string, stats)}
    else:
        stat_dict = {"{0} AP 0.50:0.95 all".format(iou): stats[0]}
    return stat_dict

def evaluate_coco_predictions(annotations_file, iou_types, predictions, verbose=False):
    cocoGt = COCO(annotation_file=annotations_file, use_ext=True)
    stat_dict = dict()
    for iou in iou_types:
        cocoDt = cocoGt.loadRes(predictions[iou], use_ext=True)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=iou, use_ext=True)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stat_dict[iou] = build_output_dict(iou, cocoEval.stats, verbose)
    return stat_dict