import gc
import torch
from sagemakercv.detection.detector import build_detection_model
from utils import (unarchive_data, 
                   get_training_world, 
                   is_sm, 
                   is_sm_dist, 
                   get_herring_world,
                   config_check)
from sagemakercv.training import make_optimizer, make_lr_scheduler
from sagemakercv.data import make_data_loader
from sagemakercv.data.utils import prefetcher
from sagemakercv.utils.checkpoint import DetectronCheckpointer
from sagemakercv.utils.runner import build_hooks, Runner
from sagemakercv.utils.runner.hooks.checkpoint import DetectronCheckpointHook
from sagemakercv.training.trainers import train_step
from sagemakercv.training.optimizers import MLPerfFP16Optimizer
import apex
from configs import cfg
from tqdm.auto import tqdm
from statistics import mean

if (torch._C, '_jit_set_profiling_executor') :
    torch._C._jit_set_profiling_executor(False)
if (torch._C, '_jit_set_profiling_mode') :
    torch._C._jit_set_profiling_mode(False)
    
cfg.merge_from_file('configs/ec2_torch_mrcnn_bs4_fp16_1x_1node.yaml')

cfg.freeze()
gc.disable()
train_coco_loader, num_iterations = make_data_loader(cfg)

device = torch.device(cfg.MODEL.DEVICE)
train_iterator = prefetcher(iter(train_coco_loader), device)

model = build_detection_model(cfg)
model.to(device)
params = []
lr = cfg.SOLVER.BASE_LR
weight_decay = cfg.SOLVER.WEIGHT_DECAY

bias_params = []
bias_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
bias_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

for key, value in model.named_parameters():
    if not value.requires_grad:
        continue
    if "bias" in key:
        bias_params.append(value)
    else:
        params.append(value)
#optimizer = make_optimizer(cfg, model)
'''optimizer = apex.optimizers.fused_sgd.FusedSGD([
                    {"params": params, "lr": lr, "weight_decay": weight_decay},
                    {"params": bias_params, "lr": bias_lr, "weight_decay": bias_weight_decay}
                ],
                lr, momentum=cfg.SOLVER.MOMENTUM)'''

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O3")            
#scheduler = make_lr_scheduler(cfg, optimizer)
#optimizer = MLPerfFP16Optimizer(optimizer, dynamic_loss_scale=True)

hooks = build_hooks(cfg)

is_fp16 = (cfg.DTYPE == "float16")
if is_fp16:
    # convert model to FP16
    model.half()

optimizer.zero_grad()
for iteration, (images, targets) in tqdm(enumerate(prefetcher(iter(train_coco_loader), device), 0)):
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    with apex.amp.scale_loss(losses, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #scheduler.step()
    
# runner = Runner(model, train_step, cfg, device, optimizer, scheduler)

# for hook in hooks:
#     runner.register_hook(hook, priority='HIGHEST' if isinstance(hook, DetectronCheckpointHook) else 'NORMAL')
    
# runner.run(train_iterator, num_iterations)

'''for i in tqdm(range(1000)):
        images, targets = next(train_iterator)
        # images = images.to(device)
        # targets = [target.to(device) for target in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.backward(losses)
        optimizer.step()
        scheduler.step()'''

'''with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('/data/profiles/test1'),
    with_stack=True
) as profiler:
    for i in tqdm(range(100)):
        images, targets = next(train_iterator)
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.backward(losses)
        optimizer.step()
        scheduler.step()
        profiler.step()'''

'''from time import time
start_time = time()
optimizer.zero_grad()
zero_grad_time = time() - start_time
loss_dict = model(images, targets)
losses = sum(loss for loss in loss_dict.values())
forward_pass_time = time() - (start_time + zero_grad_time)
losses.backward()
backward_pass_time = time() - (start_time + forward_pass_time)
optimizer.step()
scheduler.step()
step_time = time() - (start_time + backward_pass_time)
print(f"zero_grad_time: {zero_grad_time}")
print(f"forward_pass_time: {forward_pass_time}")
print(f"backward_pass_time: {backward_pass_time}")
print(f"step_time: {step_time}")'''