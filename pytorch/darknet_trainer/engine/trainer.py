import torch
from apex import amp
from statistics import mean
from data.utils import mixup_data, mixup_criterion

class Trainer(object):
    
    def __init__(self, model, loss_fn, optimizer, scheduler, mixup_alpha=0.):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.mixup_alpha = mixup_alpha
        
    def train_step(self, images, targets):
        if self.mixup_alpha>0.:
            images, y_a, y_b, lam = mixup_data(images, targets, alpha=self.mixup_alpha)
        self.optimizer.zero_grad()
        pred = self.model(images)
        if self.mixup_alpha>0.:
            loss = mixup_criterion(self.loss_fn, pred, y_a, y_b, lam)
        else:
            loss = self.loss_fn(pred, targets)
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
        '''with torch.cuda.amp.autocast():
            output = self.model(images)
            loss = self.loss_fn(output, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()'''
        self.scheduler.step()
        top_1 = (torch.argmax(pred, axis=1)==targets).double().mean()
        return loss, top_1
    
    def val_step(self, images, targets):
        pred = self.model(images)
        return float((torch.argmax(pred, axis=1)==targets).double().mean())
    
    def train_epoch(self, data_iterator, steps, log=True, log_interval=50):
        _ = self.model.train()
        loss_history = []
        top_1_history = []
        for i in range(steps):
            images, targets = next(data_iterator)
            loss, top_1 = self.train_step(images, targets)
            loss_history.append(float(loss))
            top_1_history.append(float(top_1))
            if i%log_interval==0:
                loss_mean = mean(loss_history)
                top_1_mean = mean(top_1_history)*100
                loss_history = []
                top_1_history = []
                if log:
                    print(f"Step {i}, Loss: {loss_mean}, top 1: {top_1_mean}, LR: {self.scheduler.get_lr()[0]}")
    
    def eval_epoch(self, data_iterator, steps=25):
        _ = self.model.eval()
        performance = []
        for i in range(steps):
            images, targets = next(data_iterator)
            performance.append(self.val_step(images, targets))
        return mean(performance)
        