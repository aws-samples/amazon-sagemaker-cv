import random

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
from s3fs import S3FileSystem
from configs import cfg
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import functional as F
from sagemakercv.core.structures.image_list import ImageList, to_image_list

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for i, t in enumerate(self.transforms):
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image_size = image.shape[-2:]
            image_size = (image_size[1], image_size[0])
        else:
            image_size = image.size
        size = self.get_size(image_size)
        image = F.resize(image, size)
        return image
    
class ToTensor(object):
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return F.convert_image_dtype(image, dtype=torch.float32)
        else:
            return F.to_tensor(image)

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255
        self.bgr255_indexes = None

    def __call__(self, image):
        if self.to_bgr255:
            if image.is_cuda:
                if self.bgr255_indexes is None:
                    self.bgr255_indexes = torch.tensor([2, 1, 0], dtype=torch.int64, pin_memory=True).to(device=image.device, non_blocking=True)
                image = image[self.bgr255_indexes] * 255
            else:
                image = image[[2, 1, 0]] * 255
        if not isinstance(self.mean, torch.Tensor):
            if image.is_cuda:
                self.mean = torch.tensor(self.mean, dtype=image.dtype, pin_memory=True).to(device=image.device, non_blocking=True)
                self.std = torch.tensor(self.std, dtype=image.dtype, pin_memory=True).to(device=image.device, non_blocking=True)
            else:
                self.mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device)
                self.std = torch.tensor(self.std, dtype=image.dtype, device=image.device)
            if self.mean.ndim == 1:
                self.mean = self.mean.view(-1, 1, 1)
            if self.std.ndim == 1:
                self.std = self.std.view(-1, 1, 1)
        image.sub_(self.mean)
        image.div_(self.std)
        return image

def build_transforms(cfg):
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    ops = [
              Resize(min_size, max_size),
              ToTensor(),
              normalize_transform
          ]
    transform = Compose(ops)

    return transform

class Visualizer(object):
    
    def __init__(self, model, cfg, temp_dir='.', categories=None, device='cuda'):
        self.model = model
        self.dir = temp_dir
        self.s3fs = S3FileSystem()
        self.cfg = cfg
        self.transforms = build_transforms(cfg)
        self.categories = categories
        self.device = device
        
    def get_input_image(self, src):
        if src.startswith('http'):
            response = requests.get(src)
            img = Image.open(BytesIO(response.content))
        elif src.startswith('s3'):
            local_src = os.path.join(self.dir, src.split('/')[-1])
            self.s3fs.get(src, local_src)
            img = Image.open(local_src)
        else:
            img = Image.open(src)
        return img
    
    def preprocess_image(self, image):
        '''
        Resize, rotate, and normalize image
        '''
        return to_image_list([self.transforms(image)], self.cfg.DATALOADER.SIZE_DIVISIBILITY)
    
    def postprocess_image(self, image):
        '''
        rotate and un-normalize image
        '''
        return np.fliplr(np.rot90(np.flip(torch.transpose(image, 0, 2).numpy() + self.cfg.INPUT.PIXEL_MEAN, 2), k=3)/255.)
    
    def build_image(self, image, box_list, threshold=0.9, figsize=(25, 25)):
        bboxes = box_list.bbox
        scores = box_list.extra_fields['scores']
        labels = box_list.extra_fields['labels']
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        for box, score, label in zip(bboxes, scores, labels):
            if score>=threshold:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                if self.categories:
                    caption = "{} {:.3f}".format(self.categories[int(label)], score)
                    ax.text(x1, y1, caption, size=25,
                                color='r', backgroundcolor="none")
                ax.add_patch(rect)
        plt.show()
    
    def __call__(self, image_src, threshold=0.9, figsize=(25, 25)):
        _ = self.model.eval()
        image = self.get_input_image(image_src)
        image_list = self.preprocess_image(image)
        box_list = self.model(image_list.to(self.device))[0]
        image = self.postprocess_image(image_list.tensors[0])
        self.build_image(image, box_list, threshold=threshold, figsize=figsize)
        