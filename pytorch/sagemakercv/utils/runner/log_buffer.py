# Copyright (c) Open-MMLab. All rights reserved.
from collections import OrderedDict
import torch

import numpy as np


class LogBuffer(object):

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            if torch.is_tensor(var):
                if var.device.type=='cuda':
                    var = var.cpu()
                var = var.detach().numpy()
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            if 'time' in key:
                self.output[key] = self.val_history[key][-1]
            elif 'image' not in key: # skip images
                values = np.array(self.val_history[key][-n:])
                nums = np.array(self.n_history[key][-n:])
                avg = np.sum(values * nums) / np.sum(nums)
                self.output[key] = avg
        self.ready = True