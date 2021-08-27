# Copyright (c) Open-MMLab. All rights reserved.
import time
import datetime
from .hook import Hook
from ..builder import HOOKS

class IterTimerHook(Hook):
    
    def __init__(self, interval=25):
        self.interval = interval
        
    def before_run(self, runner):
        self.start_time = datetime.datetime.now()
        if runner.rank == 0:
            runner.logger.info("Start time: {}".format(str(self.start_time)))
    
    def before_epoch(self, runner):
        self.t = time.time()
    
    def after_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval) and runner.rank == 0:
            iter_time = (time.time() - self.t)/self.interval
            runner.log_buffer.update({'time': iter_time})
            self.t = time.time()
            
    def after_run(self, runner):
        end_time = datetime.datetime.now()
        if runner.rank == 0:
            runner.logger.info("End time: {}".format(str(self.start_time)))
            runner.logger.info("Elapsed time: {}".format(str(end_time-self.start_time)))

@HOOKS.register("IterTimerHook")
def build_iter_time_hook(cfg):
    return IterTimerHook(interval=cfg.LOG_INTERVAL)