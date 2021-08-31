import torch
import apex

def train_step(images, targets, model, optimizer, scheduler, device, opt_level, grad_clip=0.0):
    optimizer.zero_grad()
    # images = images.to(device)
    # targets = [target.to(device) for target in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    if opt_level=="O4":
        optimizer.backward(losses) 
    else:
        with apex.amp.scale_loss(losses, optimizer) as scaled_loss:
            scaled_loss.backward()
    if grad_clip>0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    loss_dict['total_loss'] = losses
    return loss_dict
