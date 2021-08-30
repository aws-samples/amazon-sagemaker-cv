import os
import json
from ast import literal_eval
import torch

import importlib
import importlib.util
import sys

def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module

def unarchive_data(data_dir, target='coco.tar'):
    print("Unarchiving COCO data")
    os.system('tar -xf {0} -C {1}'.format(os.path.join(data_dir, target), data_dir))
    print(os.listdir(data_dir))

def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker

    return world

def is_sm():
    """Check if we're running inside a sagemaker training job
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if not isinstance(sm_training_env, dict):
        return False
    return True

def is_sm_dist():
    """Check if environment variables are set for Sagemaker Data Distributed
    This has not been tested
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if not isinstance(sm_training_env, dict):
        return False
    sm_training_env = literal_eval(sm_training_env)
    additional_framework_parameters = sm_training_env.get('additional_framework_parameters', None)
    if not isinstance(additional_framework_parameters, dict):
        return False
    return bool(additional_framework_parameters.get('sagemaker_distributed_dataparallel_enabled', False))

def get_herring_world():
    return {"machine_rank": 0, "number_of_processes": 8, "size": 8}

def config_check(cfg):
    """Check for incompatible settings in configuration
    """
    assert cfg.DTYPE in ["float32", "float16", "amp"], \
    f"DTYPE {cfg.DTYPE} not available. Available DTPYEs are float32, float16, and amp"
    if cfg.NHWC:
        assert cfg.MODEL.RESNETS.TRANS_FUNC.endswith("NHWC"), \
        "When using NHWC, TRANS_FUNC must be NHWC compatible, BottleneckWithFixedBatchNormNHWC"
        assert cfg.DTYPE=="float16", "NHWC currently only available with DTYPE float16"
    if cfg.DTYPE=="amp" and "AMP_Hook" not in cfg.HOOKS:
        print("Adding AMP_Hook")
        cfg.HOOKS.append("AMP_Hook")
    if cfg.DTYPE=="float16" and "FP16_Hook" not in cfg.HOOKS:
        print("Adding FP16_Hook")
        cfg.HOOKS.append("FP16_Hook")
    return
