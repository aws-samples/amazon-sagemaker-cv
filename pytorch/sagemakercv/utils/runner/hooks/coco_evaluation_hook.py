from .hook import Hook
from sagemakercv.inference.tester import test
from sagemakercv.utils.async_evaluator import init, get_evaluator, set_epoch_tag
from sagemakercv.utils.comm import master_only
from sagemakercv.utils.comm import get_world_size
from ..builder import HOOKS
from time import sleep

class COCOEvaluation(Hook):
    
    def __init__(self, interval=25, distributed=True, per_epoch=True):
        self.interval = interval
        self.distributed = distributed
        self.running_eval = False
        self.per_epoch = per_epoch
    
    def before_run(self, runner):
        init()
    
    def after_train_epoch(self, runner):
        if self.per_epoch:
            set_epoch_tag(runner.epoch)
            results = test(runner.cfg, runner.model, self.distributed)
            self.running_eval = True
    
    @master_only
    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.interval) and self.running_eval:
            evaluator = get_evaluator()
            self.running_eval = False
            all_results = {}
            for t, r in evaluator.finished_tasks().items():
                # Note: one indirection due to possibility of multiple test datasets
                # we only care about the first
                map_results = r# [0]
                bbox_map = map_results.results["bbox"]['AP']
                segm_map = map_results.results["segm"]['AP'] \
                           if runner.cfg.MODEL.MASK_ON else None
                all_results.update({ t : (bbox_map, segm_map) })
            runner.logger.info(all_results)
    
    def after_run(self, runner):
        if not self.per_epoch:
            set_epoch_tag(runner.epoch)
            results = test(runner.cfg, runner.model, self.distributed)
            self.running_eval = True
            # temp fix for async issue
            sleep(20)
        if runner.rank == 0:
            evaluator = get_evaluator()
            self.running_eval = False
            evaluator.wait_all_tasks()
            all_results = {}
            for t, r in evaluator.finished_tasks().items():
                # Note: one indirection due to possibility of multiple test datasets
                # we only care about the first
                map_results = r# [0]
                bbox_map = map_results.results["bbox"]['AP']
                segm_map = map_results.results["segm"]['AP'] \
                           if runner.cfg.MODEL.MASK_ON else None
                all_results.update({ t : (bbox_map, segm_map) })
            runner.logger.info(all_results)
    
@HOOKS.register("COCOEvaluation")
def build_coco_eval_hook(cfg):
    return COCOEvaluation(distributed=get_world_size()>1, per_epoch=cfg.TEST.PER_EPOCH_EVAL)