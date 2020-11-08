import numpy as np
import torch
import torch.nn as nn


class MemoryBank(object):
    def __init__(self, size, batch_size, dim, device):
        self.size = size
        self.batch_size = batch_size
        self.dim = dim
        self.count = 0
        self.M = torch.ones(size, dim, device=device, requires_grad=False)

    def update(self, x):
        self.M = torch.cat([x.to(self.M.device), self.M[:-len(x)]])
        self.count = min(self.count+len(x), self.size) 

    def sample_batch(self):
        if self.is_empty():
            return None
        else:
            perm = torch.randperm(min(self.count, self.size))
            idx = perm[:min(self.batch_size, len(perm))]
            return self.M[idx]

    def is_empty(self):
        return self.count == 0

    def __getitem__(self, index):
        if self.is_empty():
            return None
        else:
            return self.M[min(self.count-1, index)]