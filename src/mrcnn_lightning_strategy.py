import torch
import os
from pytorch_lightning.core.optimizer import _configure_schedulers_automatic_opt, _set_scheduler_opt_idx, _validate_scheduler_api, _configure_schedulers_manual_opt

from pytorch_lightning.strategies import Strategy
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers
from pytorch_lightning.trainer.states import TrainerFn

def custom_configure_optimizers(optim_conf):
    optimizers, lr_schedulers, optimizer_frequencies = [], [], []
    monitor = None
    optimizers = [optim_conf]
    return optimizers, lr_schedulers, optimizer_frequencies, monitor

def custom_init_optimizers_and_lr_schedulers(
    model: "pl.LightningModule",
):
    """Calls `LightningModule.configure_optimizers` and parses and validates the output."""
    assert model.trainer is not None
    optim_conf = model.trainer._call_lightning_module_hook("configure_optimizers", pl_module=model)

    if optim_conf is None:
        rank_zero_warn(
            "`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer",
        )
        optim_conf = _MockOptimizer()

    optimizers, lr_schedulers, optimizer_frequencies, monitor = custom_configure_optimizers(optim_conf)
    lr_scheduler_configs = (
        _configure_schedulers_automatic_opt(lr_schedulers, monitor)
        if model.automatic_optimization
        else _configure_schedulers_manual_opt(lr_schedulers)
    )
    _set_scheduler_opt_idx(optimizers, lr_scheduler_configs)
    _validate_scheduler_api(lr_scheduler_configs, model)
    return optimizers, lr_scheduler_configs, optimizer_frequencies


class MRCNNLightningStrategy(Strategy):
    
    def __init__(self, 
        device = "cuda",
        accelerator = None,
        checkpoint_io = None,
        precision_plugin = None,):
        
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self._root_device = torch.device(device)
        self.global_rank = os.environ.get("RANK", 0)
        self.root_device = torch.device("cuda:0")
        
#     def backward(self, closure_loss, * args, ** kwargs):
#         pass
    
#     def batch_to_device(self, batch, device = None, dataloader_idx = 0):
#         pass
    
#     def connect(self, model):
#         pass
    
#     def dispatch(self, trainer):
#         pass
    
#     def lightning_module_state_dict(self, ):
#         pass
    
#     def model_sharded_context(self, ):
#         yield
    
#     def on_predict_end(self, ):
#         pass
    
#     def on_predict_start(self, ):
#         pass
    
#     def on_test_end(self, ):
#         pass
        
#     def on_test_start(self, ):
#         pass
    
#     def on_train_batch_start(self, batch, batch_idx, dataloader_idx = 0):
#         pass
    
#     def on_train_end(self, ):
#         pass
    
#      def on_train_start(self, ):
#         pass
    
#     def on_validation_end(self, ):
#         pass
    
#     def on_validation_start(self, ):
#         pass
    
#     def optimizer_state(self, optimizer):
#         pass
    
#     def optimizer_step(self, optimizer, opt_idx, closure, model = None, ** kwargs):
#         pass
    
#     def post_backward(self, closure_loss):
#         pass
    
#     def post_dispatch(self, trainer):
#         pass
    
#     def pre_backward(self, closure_loss):
#         pass
    
#     def predict_step(self, * args, ** kwargs):
#         pass
    
#     def process_dataloader(self, dataloader):
#         pass
    
#     def reduce_boolean_decision(self, decision):
#         pass
    
#     def remove_checkpoint(self, filepath):
#         pass
    
#     def save_checkpoint(self, checkpoint, filepath, storage_options = None):
#         pass
    
#     def setup(self, trainer):
#         pass
    
#     def setup_environment(self, ):
#         pass
    
    def setup_optimizers(self, trainer):
        if trainer.state.fn not in (TrainerFn.FITTING, TrainerFn.TUNING):
            return
        self.optimizers, self.lr_scheduler_configs, self.optimizer_frequencies = custom_init_optimizers_and_lr_schedulers(
            self.lightning_module
        )
    
#     def setup_precision_plugin(self, ):
#         pass
    
#     def teardown(self, ):
#         pass
    
#     def test_step(self, * args, ** kwargs):
#         pass
    
#     def training_step(self, * args, ** kwargs):
#         pass
    
#     def validation_step(self, * args, ** kwargs):
#         pass
    
    def all_gather(self, tensor, group = None, sync_grads = False):
        pass
    
    def barrier(self, name = None):
        pass
    
    def broadcast(self, obj, src = 0):
        return obj
#         pass
    
    def is_global_zero(self, ):
        return True
#         pass
    
    def model_to_device(self, ):
        assert self.model is not None, "self.model must be set before self.model.to()"
        self.model.to(self.root_device)
    
    def reduce(self, tensor, group = None, reduce_op = 'mean'):
        pass
    
    def root_device(self) -> torch.device:
        return self._root_device