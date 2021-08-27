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

import matplotlib.pyplot as plt
import colorsys
import random
import numpy as np
from matplotlib import patches
from sagemakercv.data.coco.coco_labels import coco_categories
import tensorflow as tf

def random_colors(N, bright=True):
    '''
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    '''
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def add_boxes(boxes, 
              ax, 
              scores=None,
              class_ids=None, 
              class_names=None,
              threshold=0):
    if threshold>0:
        assert scores!=None
        N = tf.where(scores>threshold).shape[0]
    else:
        N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        return ax
    colors = random_colors(N)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        # Label
        if scores!=None:
            class_id = int(class_ids[i])
            score = scores[i] if scores is not None else None
            label = class_names[class_id] if class_names is not None else class_id
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
            ax.text(x1, y1 + 8, caption, size=15,
                    color=color, backgroundcolor="none")
    return ax

def restore_image(image, 
                  image_info,
                  mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225]):
    image_info = tf.cast(image_info, tf.int32)
    image = image[:image_info[0], :image_info[1]]
    image = tf.clip_by_value((image * std + mean), 0, 1) * 255
    return image

def build_image(image,
                boxes=None,
                scores=None,
                class_ids=None, 
                class_names=None,
                threshold=0,
                figsize=(10, 10), 
                title=""):
    # get original image
    if tf.is_tensor(image):
        image = image.numpy()
    
    fig, ax = plt.subplots(1, figsize=figsize)
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    if boxes!=None:
        ax = add_boxes(boxes, ax, scores, class_ids, class_names, threshold)
    plt.imshow(image.astype(np.uint8))
    fig.tight_layout(pad = 0.0)
    fig.canvas.draw()
    plt.close(fig)
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
