# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
# from torchvision.io.image import ImageReadMode
import torch.multiprocessing as mp

from sagemakercv.core.structures.image_list import ImageList, to_image_list
from sagemakercv.core.structures.bounding_box import BoxList
from sagemakercv.core.structures.segmentation_mask import SegmentationMask
from sagemakercv.core.structures.keypoint import PersonKeypoints

from awsio.python.lib.io.s3.s3dataset import S3BaseClass
import _pywrap_s3_io

from PIL import Image
import io
import logging

import os
import pickle
import numpy as np

min_keypoints_per_image = 10

# keep PIL from logging exif data
logging.getLogger('PIL').setLevel(logging.WARNING)

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, pkl_ann_file=None
    ):
        if pkl_ann_file and os.path.exists(pkl_ann_file):
            with open(pkl_ann_file, "rb") as f:
                unpickled = pickle.loads(f.read())
                self.root = root
                self.coco = unpickled["coco"]
                self.ids = unpickled["ids"]
                self.transform = None
                self.target_transform = None
                self.transforms = None
        else:
            super(COCODataset, self).__init__(root, ann_file)
            # sort indices for reproducible results
            self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self._hybrid = False
        
    def build_target(self, anno, img_size, pin_memory=False):
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.tensor(boxes, dtype=torch.float32, pin_memory=pin_memory).reshape(-1, 4) # guard against no boxes
        target = BoxList(boxes, img_size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes, dtype=torch.float32, pin_memory=pin_memory)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img_size, pin_memory=pin_memory)
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img_size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        return target

    def __getitem__(self, idx):
        if self._hybrid:
            # return decoded raw image as byte tensor
            #orig_img, _ = super(COCODataset, self).__getitem__(idx)
            #orig_img_tensor = torchvision.transforms.functional.to_tensor(orig_img)
            img = torchvision.io.read_image(self.get_raw_img_info(idx), 3) #ImageReadMode.RGB)
            #print("orig_img.size = %s, img.shape = %s, orig_img_tensor.shape = %s" % (str(orig_img.size), str(img.shape), str(orig_img_tensor.shape)))
            target = self.get_target(idx)
            return img, target, idx
        else:
            img, anno = super(COCODataset, self).__getitem__(idx)
            target = self.build_target(anno, img.size)

            #orig_img, _ = super(COCODataset, self).__getitem__(idx)
            #orig_img_tensor = torchvision.transforms.functional.to_tensor(orig_img)
            #img = torchvision.io.read_image(self.get_raw_img_info(idx), ImageReadMode.RGB)
            #print("orig_img.size = %s, img.shape = %s, orig_img_tensor.shape = %s" % (str(orig_img.size), str(img.shape), str(orig_img_tensor.shape)))
            #target = self.get_target(idx)

            if self._transforms is not None:
                img, target = self._transforms(img, target)
            #print("img.shape = %s, target = %s, img.sum = %f" % (str(img.shape), str(target), img.float().sum()))
            return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_raw_img_info(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return os.path.join(self.root, path)

    def get_target(self, index, pin_memory=False):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        img_size = (self.coco.imgs[img_id]["width"], self.coco.imgs[img_id]["height"])
        return self.build_target(anno, img_size, pin_memory=pin_memory)

class S3COCODataset(S3BaseClass, torchvision.datasets.coco.CocoDetection):
    
    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None, pkl_ann_file=None):
        print("Reading objects from S3")
        S3BaseClass.__init__(self, root)
        print("{} objects found".format(len(self.urls_list)))
        if pkl_ann_file and os.path.exists(pkl_ann_file):
            with open(pkl_ann_file, "rb") as f:
                unpickled = pickle.loads(f.read())
                self.root = root
                self.coco = unpickled["coco"]
                self.ids = unpickled["ids"]
                self.transform = None
                self.target_transform = None
                self.transforms = None
        else:
            torchvision.datasets.coco.CocoDetection.__init__(self, root, ann_file)
            self.ids = sorted(self.ids)
        self.handler = None
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        
    def build_target(self, anno, img_size, pin_memory=False):
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.tensor(boxes, dtype=torch.float32, pin_memory=pin_memory).reshape(-1, 4) # guard against no boxes
        target = BoxList(boxes, img_size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes, dtype=torch.float32, pin_memory=pin_memory)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img_size, pin_memory=pin_memory)
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img_size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        return target
    
    def _load_image(self, image_id: int):
        if self.handler == None:
            self.handler = _pywrap_s3_io.S3Init()
        filename = os.path.join(self.root, self.coco.loadImgs(image_id)[0]["file_name"])
        fileobj = self.handler.s3_read(filename)
        return Image.open(io.BytesIO(fileobj)).convert("RGB")
    
    def _load_target(self, image_id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(image_id))
    
    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = self._load_image(image_id)
        anno = self._load_target(image_id)
        target = self.build_target(anno, img.size)
        if self._transforms is not None:
                img, target = self._transforms(img, target)
        return img, target, idx
        
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    def get_raw_img_info(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return os.path.join(self.root, path)

    def get_target(self, index, pin_memory=False):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        img_size = (self.coco.imgs[img_id]["width"], self.coco.imgs[img_id]["height"])
        return self.build_target(anno, img_size, pin_memory=pin_memory)

def load_file(path):
    with open(path, 'rb') as f:
        raw_image = np.frombuffer(f.read(), dtype=np.uint8)
    return raw_image

class COCODALIBatchIterator(object):
    def __init__(self, batch_size, batch_sampler, dataset):
        self.batch_size = batch_size
        self.batch_sampler = batch_sampler
        self.batch_sampler_iter = None
        self.num_samples = len(self.batch_sampler)
        self.dataset = dataset

    def __iter__(self):
        self.batch_sampler_iter = iter(self.batch_sampler)
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        batch = [(load_file(self.dataset.get_raw_img_info(index)),index) for index in next(self.batch_sampler_iter)]
        raw_images, indices = tuple(zip(*batch))
        raw_images, indices = list(raw_images), list(indices)
        nshort = self.batch_size - len(raw_images)
        if nshort > 0:
            # DALI pipeline dislikes incomplete batches, so pad
            raw_images = raw_images + [raw_images[0]]*nshort
            indices = indices + [-1]*nshort
        return [raw_images, np.asarray(indices)]


class HybridDataLoader(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible, shapes):
        dataset._hybrid = True
        assert(dataset._transforms is None), "dataset.transforms must be None when hybrid dataloader is selected"
        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
        )
        self.iter = None
        self.transforms = transforms
        self.size_divisible = size_divisible
        self.shapes = shapes

    def __iter__(self):
        self.iter = iter(self.data_loader)
        return self

    def __len__(self):
        return len(self.data_loader)

    def __next__(self):
        images, targets = [], []
        raw_images, raw_targets, idxs = next(self.iter)
        for raw_image, raw_target in zip(raw_images, raw_targets):
            image = raw_image.cuda()
            image, target = self.transforms(image, raw_target)
            images.append( image )
            targets.append( target )
            #print("image.shape = %s, target = %s, image.sum = %f" % (str(image.shape), str(target), image.float().sum()))
        images = to_image_list(images, self.size_divisible, self.shapes)
        #print("images.tensors.size = %s, images.image_sizes = %s" % (str(images.tensors.size()), str(images.image_sizes)))
        return images, targets, idxs

done = mp.Event()

def hybrid_loader_worker(rank, size, batch_sampler, dataset, txbufs, q):
    j = 0
    for i, batch in enumerate(batch_sampler):
        if i % size == rank:
            metadata = []
            for idx, txbuf in zip(batch, txbufs[j]):
                img = torchvision.io.read_image(dataset.get_raw_img_info(idx), 3) # ImageReadMode.RGB)
                txbuf[:img.numel()].copy_(img.flatten())
                metadata.append( (list(img.size()), idx) )
            q.put( (j, metadata) )
            j = (j + 1) % 3
    done.wait()

class HybridDataLoader2(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible, shapes):
        dataset._hybrid = True
        assert(dataset._transforms is None), "dataset._transforms must be None when hybrid dataloader is selected"
        self.batch_size = batch_size
        self.batch_sampler = batch_sampler
        self.dataset = dataset
        self.i = 0
        self.length = len(self.batch_sampler)
        self.transforms = transforms
        self.size_divisible = size_divisible
        self.shapes = shapes
        self.num_workers=cfg.DATALOADER.NUM_WORKERS
        maxsize = cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST
        self.workers, self.queues, self.txbufs = [], [], []
        for worker in range(self.num_workers):
            txbuf = [torch.empty(size=[batch_size,3*maxsize*maxsize], dtype=torch.uint8).pin_memory() for _ in range(3)]
            for t in txbuf: t.share_memory_()
            self.txbufs.append( txbuf )
            q = mp.Queue(maxsize=1)
            self.queues.append( q )
            p = mp.Process(target=hybrid_loader_worker, args=(worker,self.num_workers,batch_sampler,dataset,txbuf,q,))
            self.workers.append( p )
            p.start()

    def __del__(self):
        for p in self.workers:
            p.terminate()

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.i < self.length:
            worker = self.i % self.num_workers
            p, q, txbufs = self.workers[worker], self.queues[worker], self.txbufs[worker]
            images, targets, idxs = [], [], []
            j, metadata = q.get()
            for txbuf, (img_size, idx) in zip(txbufs[j], metadata):
                numel = img_size[0] * img_size[1] * img_size[2]
                raw_image = txbuf[:numel].reshape(img_size)
                raw_image = raw_image.to(device='cuda', non_blocking=True)
                raw_target = self.dataset.get_target(idx, pin_memory=True)
                image, target = self.transforms(raw_image, raw_target)
                images.append( image )
                targets.append( target )
                idxs.append( idx )
            images = to_image_list(images, self.size_divisible, self.shapes)
            self.i += 1
            return images, targets, idxs
        else:
            done.set()
            raise StopIteration()

class HybridDataLoader3(object):
    def __init__(self, cfg, is_train, batch_size, batch_sampler, dataset, collator, transforms, size_divisible, shapes):
        dataset._hybrid = True
        assert(dataset._transforms is None), "dataset._transforms must be None when hybrid dataloader is selected"
        self.batch_size = batch_size
        self.length = len(batch_sampler)
        self.batch_sampler = iter(batch_sampler)
        self.dataset = dataset
        self.transforms = transforms
        self.size_divisible = size_divisible
        self.shapes = shapes

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        images, targets, idxs = [], [], []
        for idx in next(self.batch_sampler):
            raw_image = torchvision.io.read_image(self.dataset.get_raw_img_info(idx), 3).pin_memory().to(device='cuda', non_blocking=True) # ImageReadMode.RGB).pin_memory().to(device='cuda', non_blocking=True)
            raw_target = self.dataset.get_target(idx, pin_memory=True)
            image, target = self.transforms(raw_image, raw_target)
            images.append( image )
            targets.append( target )
            idxs.append( idx )
        images = to_image_list(images, self.size_divisible, self.shapes)
        return images, targets, idxs