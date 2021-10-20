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

import functools
import os
import sys
from ast import literal_eval
import json

__all__ = ["MPI_local_rank", "MPI_rank", "MPI_size", "MPI_rank_and_size", "MPI_is_distributed", "is_herring"]

def MPI_is_distributed():
    """Return a boolean whether a distributed training/inference runtime is being used.
    :return: bool
    """
    if all([var in os.environ for var in ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]]):
        return True
    elif all([var in os.environ for var in ["SLURM_PROCID", "SLURM_NTASKS"]]):
        return True
    else:
        return False

def MPI_local_rank():
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
    elif "SLURM_LOCALID" in os.environ:
        return int(os.environ.get("SLURM_LOCALID"))
    else:
        return 0

def MPI_local_size():
    return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))

def MPI_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))

def MPI_size():
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

def MPI_rank_and_size():
    return MPI_rank(), MPI_size()

def get_dist_info():
    rank = MPI_rank()
    local_rank = MPI_local_rank()
    size = MPI_size()
    local_size = MPI_local_size()
    return rank, local_rank, size, local_size

def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = MPI_rank()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper


# Source: https://github.com/horovod/horovod/blob/c3626e/test/common.py#L25
def mpi_env_MPI_rank_and_size():
    """Get MPI rank and size from environment variables and return them as a
    tuple of integers.
    Most MPI implementations have an `mpirun` or `mpiexec` command that will
    run an MPI executable and set up all communication necessary between the
    different processors. As part of that set up, they will set environment
    variables that contain the rank and size of the MPI_COMM_WORLD
    communicator. We can read those environment variables from Python in order
    to ensure that `hvd.rank()` and `hvd.size()` return the expected values.
    Since MPI is just a standard, not an implementation, implementations
    typically choose their own environment variable names. This function tries
    to support several different implementation, but really it only needs to
    support whatever implementation we want to use for the TensorFlow test
    suite.
    If this is not running under MPI, then defaults of rank zero and size one
    are returned. (This is appropriate because when you call MPI_Init in an
    application not started with mpirun, it will create a new independent
    communicator with only one process in it.)

    Source: https://github.com/horovod/horovod/blob/c3626e/test/common.py#L25
    """
    rank_env = 'PMI_RANK SLURM_PROCID OMPI_COMM_WORLD_RANK'.split()
    size_env = 'PMI_SIZE SLURM_NTASKS OMPI_COMM_WORLD_SIZE'.split()

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)

    # Default to rank zero and size one if there are no environment variables
    return 0, 1

def is_sm_dist():
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