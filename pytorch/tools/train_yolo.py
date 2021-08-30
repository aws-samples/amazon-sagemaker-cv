import sys
import os
from statistics import mean
import logging

import torch
import apex
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
#from torch.nn.parallel import DistributedDataParallel as DDP

from sagemakercv.data import make_data_loader
from sagemakercv.data.utils import prefetcher
from sagemakercv.detection.backbone.darknet import build_darknet_backbone
from sagemakercv.detection.dense_heads.yolo_head import build_yolo_head
from sagemakercv.detection.detector.yolo_detector import Yolov3Detector
from sagemakercv.training import make_optimizer, make_lr_scheduler
from sagemakercv.utils.runner import build_hooks, Runner
from configs import cfg

if (torch._C, '_jit_set_profiling_executor') :
    torch._C._jit_set_profiling_executor(False)
if (torch._C, '_jit_set_profiling_mode') :
    torch._C._jit_set_profiling_mode(False)
    
if int(os.environ.get("RANK", 0))==0:
    logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', 
            level=logging.DEBUG,
            stream=sys.stdout)
    logger = logging.getLogger("main_process_logger")
else:
    logger = None

cfg.merge_from_file('configs/st_yolo.yaml')

rank = int(os.environ.get("RANK", 0))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
dist.init_process_group(
        backend="nccl", init_method="env://"
    )

train_coco_loader, num_iterations = make_data_loader(cfg)
device = torch.device(cfg.MODEL.DEVICE)
train_iterator = prefetcher(iter(train_coco_loader), device)
backbone = build_darknet_backbone(cfg)

state_dict = torch.load('/workspace/data/weights/epoch_89.ph')
state_dict = {key.replace('module.', ''): val for key, val in state_dict.items() if '.fc.' not in key}

# Fix this
state_dict = {key.replace('1.weight', 'bn.weight'): val for key, val in state_dict.items()}
state_dict = {key.replace('1.running_mean', 'bn.running_mean'): val for key, val in state_dict.items()}
state_dict = {key.replace('1.bias', 'bn.bias'): val for key, val in state_dict.items()}
state_dict = {key.replace('1.running_var', 'bn.running_var'): val for key, val in state_dict.items()}
state_dict = {key.replace('1.num_batches_tracked', 'bn.num_batches_tracked'): val for key, val in state_dict.items()}
state_dict = {key.replace('0.weight', 'conv.weight'): val for key, val in state_dict.items()}

backbone.backbone.load_state_dict(state_dict)

#for param in backbone.backbone.parameters():
#    param.requires_grad = False

head = build_yolo_head(cfg)
model = Yolov3Detector(backbone, head, cfg)
# state_dict = torch.load("/workspace/sagemakercv/pytorch/tools/initial_training.pt" )
# state_dict = {key.replace('module.', ''): val for key, val in state_dict.items()}
# model.load_state_dict(state_dict)

_ = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
# optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=0.001, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
# optimizer = apex.optimizers.FusedSGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
# optimizer = apex.optimizers.FusedNovoGrad(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2))
scheduler = make_lr_scheduler(cfg, optimizer)

model = DDP(model, delay_allreduce=True)

model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")

def train_step(images, targets, model, optimizer, scheduler, device, dtype, grad_clip=0.0):
    optimizer.zero_grad()
    losses, parsed_targets, feature_maps = model(images, targets)
    with apex.amp.scale_loss(losses['total_loss'], optimizer) as scaled_loss:
        scaled_loss.backward()
    if grad_clip>0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    return losses

hooks = build_hooks(cfg)

runner = Runner(model, train_step, cfg, device, optimizer, scheduler, logger=logger)
for hook in hooks:
    runner.register_hook(hook)
runner.run(train_iterator, num_iterations)

'''is_main_rank = rank==0
total_loss_history = []
box_loss_history = []
class_loss_history = []
#object_loss_history = []
for i in range(cfg.SOLVER.MAX_ITER):
    images, targets = next(train_iterator)
    losses = train_step(images, targets, model, optimizer, scheduler, device)
    total_loss_history.append(float(losses['total_loss']))
    box_loss_history.append(float(losses['box_loss']))
    class_loss_history.append(float(losses['class_loss']))
    #object_loss_history.append(float(losses['object_loss']))
    if i%100==0:
        total_loss_mean = mean(total_loss_history)
        box_loss_mean = mean(box_loss_history)
        class_loss_mean = mean(class_loss_history)
        #object_loss_mean = mean(object_loss_history)
        total_loss_history = []
        box_loss_history = []
        class_loss_history = []
        #object_loss_history = []
        if is_main_rank:
            print("step: {0} total: {1:.4f} box: {2:.4f} class: {3:.4f}".format(i, float(total_loss_mean),
                                                                                             float(box_loss_mean),
                                                                                             float(class_loss_mean)))'''

if rank==0:
    torch.save(model.state_dict(), os.path.join("initial_training_0829.pt"))
