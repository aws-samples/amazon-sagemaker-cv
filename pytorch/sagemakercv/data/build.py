# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import bisect
import copy
import logging

import torch.utils.data
from sagemakercv.utils.comm import get_world_size
from sagemakercv.utils.imports import import_file
from sagemakercv.utils.registry import Registry

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator
from .transforms import build_transforms

DATASETS = Registry()

def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    total_datasets_size = 0
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        total_datasets_size += len(dataset)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets, total_datasets_size

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset], total_datasets_size


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0, random_number_generator=None,
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter, random_number_generator,
        )
    return batch_sampler

@DATASETS.register("COCO")
def make_coco_dataloader(cfg, is_train=True, is_distributed=False, start_iter=0, random_number_generator=None):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    is_fp16 = (cfg.DTYPE == "float16") or (cfg.DTYPE == "amp" and cfg.SOLVER.OPT_LEVEL in ['O2', 'O3'])
    transforms = build_transforms(cfg, is_train, is_fp16)
    args = dict()
    args['root'] = cfg.INPUT.TRAIN_INPUT_DIR if is_train else cfg.INPUT.VAL_INPUT_DIR
    args['ann_file'] = cfg.INPUT.TRAIN_ANNO_DIR if is_train else cfg.INPUT.VAL_ANNO_DIR
    args["remove_images_without_annotations"] = is_train
    args["transforms"] = transforms
    dataset = D.COCODataset(**args)
    epoch_size = len(dataset)
    iterations_per_epoch = epoch_size // images_per_batch + 1
    sampler = make_data_sampler(dataset, is_train, is_distributed)
    batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, random_number_generator,
        )
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
        )
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        return data_loader, iterations_per_epoch
    return [data_loader]

@DATASETS.register("PathsCatalogLoader")
def paths_catalog_dataloader(cfg, is_train=True, is_distributed=False, start_iter=0, random_number_generator=None):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "configs.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    is_fp16 = (cfg.DTYPE == "float16") or (cfg.DTYPE == "amp" and cfg.SOLVER.OPT_LEVEL in ['O2', 'O3'])
    transforms = build_transforms(cfg, is_train, is_fp16)
    datasets, epoch_size = build_dataset(dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, random_number_generator,
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        iterations_per_epoch = epoch_size // images_per_batch + 1
        return data_loaders[0], iterations_per_epoch
    return data_loaders

def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, random_number_generator=None):
    return DATASETS[cfg.INPUT.DATALOADER](cfg, 
                                          is_train=is_train, 
                                          is_distributed=is_distributed, 
                                          start_iter=start_iter, 
                                          random_number_generator=random_number_generator)

def prefetcher(load_iterator, device):
    prefetch_stream = torch.cuda.Stream()
    pad_batches = []

    def _prefetch():
        try:
            # I'm not sure why the trailing _ is necessary but the reference used
            # "for i, (images, targets, _) in enumerate(data_loader):" so I'll keep it.
            images, targets, _ = next(load_iterator)
        except StopIteration:
            return None, None

        with torch.cuda.stream(prefetch_stream):
            # TODO:  I'm not sure if the dataloader knows how to pin the targets' datatype.
            targets = [target.to(device, non_blocking=True) for target in targets]
            images = images.to(device, non_blocking=True)

        return images, targets

    next_images, next_targets = _prefetch()

    while next_images is not None:
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        current_images, current_targets = next_images, next_targets
        next_images, next_targets = _prefetch()
        yield current_images, current_targets