import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def create_color_map(gt, labels2train, labels2palette):
    h, w = gt.shape[-2:]
    color_map = torch.ones((h, w, 3), dtype=torch.uint8)
    for class_id in labels2train.keys():
        color_value = torch.tensor(np.array(labels2palette[class_id]), dtype=torch.uint8)
        color_map[gt == labels2train[class_id]] = color_value
    return color_map.permute(2, 0, 1)