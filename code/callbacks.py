import os
import pickle, gzip
import torch
import pytorch_lightning as pl
from time import time
from pathlib import Path
import shutil

import smdebug.pytorch as smd
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig
from smdebug.core.collection import CollectionKeys
from smdebug.core.config_constants import DEFAULT_CONFIG_FILE_PATH

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

class PlSageMakerLogger(pl.Callback):
    
    def __init__(self, frequency=100):
        self.frequency=frequency
        self.step = 0
        self.epoch = 0
    
    def on_train_epoch_start(self, trainer, module, *args, **kwargs):
        self.inner_step = 1
        self.epoch += 1
        self.step_time_start = time()
    
    @pl.utilities.rank_zero_only
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.inner_step%self.frequency==0:
            logs = ["Step: {}".format(self.inner_step),
                    "LR: {0:.4f}".format(float(trainer.model.scheduler.get_lr()[0]))]
            for key,value in trainer.logged_metrics.items():
                logs.append("{0}: {1:.4f}".format(key, float(value)))
            step_time_end = time()
            logs.append("Step time: {0:.2f} milliseconds".format((step_time_end - self.step_time_start)/self.frequency * 1000))
            print(' '.join(logs))
            '''print("Step : {} of epoch {}".format(self.inner_step, self.epoch))
            print("Training Losses:")
            print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                            for key,value in trainer.logged_metrics.items()]))
            step_time_end = time()
            print("Step time: {0:.2f} milliseconds".format((step_time_end - self.step_time_start)/self.frequency * 1000))'''
            self.step_time_start = step_time_end
        self.inner_step += 1
        self.step += 1
        
    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer, module, *args, **kwargs):
        print("Validation")
        print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                        for key,value in trainer.logged_metrics.items() if 'val' in key]))

class ProfilerCallback(pl.Callback):
    
    def __init__(self, start_step=200, num_steps=25, output_dir='/opt/ml/checkpoints/profiler/'):
        super().__init__()
        self.__dict__.update(locals())
        self.step = 0
        self.profiler = torch.profiler.profile(activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                                    schedule=torch.profiler.schedule(wait=5,
                                                                     warmup=5,
                                                                     active=self.num_steps),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, "tensorboard")),
                                    with_stack=True)
    
    def on_train_batch_start(self, trainer, module, *args, **kwargs):
        if self.step==self.start_step:
            self.profiler.__enter__()
    
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.step>=self.start_step and self.step<=self.start_step + self.num_steps:
            self.profiler.step()
        if self.step==self.start_step + self.num_steps:
            self.profiler.__exit__(None, None, None)
        self.step += 1

class SMDebugCallback(pl.Callback):
    def __init__(self, out_dir=None,
                       export_tensorboard=False,
                       tensorboard_dir=None,
                       dry_run=False,
                       reduction_config=None,
                       save_config=None,
                       include_regex=None,
                       include_collections=None,
                       save_all=False,
                       include_workers="one",
                    ):
        super().__init__()
        self.__dict__.update(locals())
        if out_dir:
            assert not Path(out_dir).exists()
        
    def on_fit_start(self, trainer, pl_module, stage=None):
        if Path(DEFAULT_CONFIG_FILE_PATH).exists():
            # smd.Hook.register_hook(pl_module.model, pl_module.criterion)
            hook = smd.Hook.create_from_json_file()
        else:
            hook = smd.Hook(out_dir=self.out_dir,
                            export_tensorboard=self.export_tensorboard,
                            tensorboard_dir=self.tensorboard_dir,
                            reduction_config=self.reduction_config,
                            save_config=self.save_config,
                            include_regex=self.include_regex,
                            include_collections=self.include_collections,
                            save_all=self.save_all,
                            include_workers=self.include_workers)
        hook.register_module(pl_module)
        hook.register_loss(pl_module.criterion)