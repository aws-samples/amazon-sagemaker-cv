import torch
import numpy as np

class Prefetcher(object):
    
    def __init__(self, data_loader, device, NHWC=False, fp16=False):
        self.device = device
        self.data_loader = data_loader
        self.load_iterator = iter(self.data_loader)
        self.prefetch_stream = torch.cuda.Stream()
        self.next_images, self.next_targets = self._prefetch()
        self.NHWC = NHWC
        self.fp16 = fp16
    
    def _prefetch(self):
        try:
            images, targets = next(self.load_iterator) 
        except StopIteration:
            self.load_iterator = iter(self.data_loader)
            images, targets = next(self.load_iterator) 
        with torch.cuda.stream(self.prefetch_stream):
            targets = targets.to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)
        return images, targets
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        current_images, current_targets = self.next_images, self.next_targets
        self.next_images, self.next_targets = self._prefetch()
        if self.NHWC:
            current_images = current_images.to(memory_format=torch.channels_last)
        if self.fp16:
            current_images = current_images.to(dtype=torch.float16)
        return current_images, current_targets
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)