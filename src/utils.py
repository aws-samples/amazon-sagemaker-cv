import os
import json
from ast import literal_eval

import importlib
import importlib.util
import sys
import numpy as np

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
    if os.environ.get('SM_TRAINING_ENV', None) is None:
        return False
    return True

def is_smddp():
    """Check if environment variables are set for Sagemaker Data Distributed
    This has not been tested
    """
    sm_training_env = os.environ.get('SM_TRAINING_ENV', None)
    if sm_training_env is None:
        return False
    sm_training_env = json.loads(sm_training_env)
    additional_framework_parameters = sm_training_env.get('additional_framework_parameters', None)
    if not isinstance(additional_framework_parameters, dict):
        return False
    return bool(additional_framework_parameters.get('sagemaker_distributed_dataparallel_enabled', False))

def get_rank():
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return 0
    
def get_world_size():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return 1
