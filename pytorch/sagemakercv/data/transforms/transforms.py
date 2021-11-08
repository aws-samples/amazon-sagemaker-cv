# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for i, t in enumerate(self.transforms):
            image, target = t(image, target)
        return image, target

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

    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            image_size = image.shape[-2:]
            image_size = (image_size[1], image_size[0])
        else:
            image_size = image.size
        size = self.get_size(image_size)
        image = F.resize(image, size)
        if isinstance(image, torch.Tensor):
            image_size = image.shape[-2:]
            image_size = (image_size[1], image_size[0])
        else:
            image_size = image.size
        target = target.resize(image_size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            return F.convert_image_dtype(image, dtype=torch.float32), target
        else:
            return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255
        self.bgr255_indexes = None

    def __call__(self, image, target):
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
        return image, target

class ToHalf(object):
    def __call__(self, image, target):
        return F.convert_image_dtype(image, dtype=torch.float16), target

class ToFloat(object):
    def __call__(self, image, target):
        return F.convert_image_dtype(image, dtype=torch.float32), target
