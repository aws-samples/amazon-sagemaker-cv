# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
import torch

import os

#from mlperf_logging.mllog import constants

from sagemakercv.data.build import make_data_loader
from .inference import inference
from sagemakercv.utils.miscellaneous import mkdir
#from sagemakercv.utils.mlperf_logger import log_event
from sagemakercv.utils.comm import synchronize

_first_test = True

def test(cfg, model, distributed):
    #if distributed:
    #    model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    dataset_names = [f"evaluation_dataset_{i}" for i in range(len(data_loaders_val))] #cfg.DATASETS.TEST
    output_folders = [None] * len(dataset_names)
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    global _first_test
    if _first_test:
        #log_event(key=constants.EVAL_SAMPLES, value=len(data_loaders_val))
        _first_test = False

    results = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        # Note: this synchronize() would break async results by not allowing them
        # to actually be async
        # synchronize()
        results.append(result)
    return results

