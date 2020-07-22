import os
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid


def fast_hist(pred, gt, num_classes):
    k = (gt >= 0) & (gt < num_classes)
    return torch.bincount(num_classes * pred[k].type(torch.int32) + gt[k], minlength=num_classes**2).view(num_classes, num_classes)

def per_class_iou(hist):
    return torch.diag(hist).float() / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).float()

def pixel_acc(hist):
    return torch.diag(hist).sum().float() / hist.sum().float()

def compute_iou(pred, gt, num_classes):
    cm = torch.zeros(num_classes, num_classes, device=gt.device)
    for i in range(len(pred)):
        cm += fast_hist(pred[i], gt[i], num_classes)
    classwise_iou = per_class_iou(cm)
    pix_acc = pixel_acc(cm)
    return pix_acc, classwise_iou

def inf_iter(dataloader):
    while True:
        for batch_idx, data in enumerate(dataloader):
            yield batch_idx, data
            
def evaluate_segmentation_batch(inf_dataloader, seg_model, seg_loss, num_classes, device):
    _, data = next(inf_dataloader)
    image, gt, name = data['image'], data['gt'], data['name']
    image = image.to(device)
    gt = gt.to(device)
    output = seg_model(image)['out']
    output = F.interpolate(output, gt.shape[-2:], mode='bilinear', align_corners=False)
    loss = seg_loss(output, gt).item()
    pred = output.argmax(1)
    pixel_accuracy, class_iou = compute_iou(pred, gt, num_classes)
    return {'loss': loss, 
            'pixel_accuracy': pixel_accuracy.cpu().tolist(),
            'classwise_iou': class_iou.cpu().tolist()}
        
def evaluate_segmentation_set(inf_dataloader, set_size, seg_model, seg_loss, num_classes, device):
    loss = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    for i in range(set_size):
        _, data = next(inf_dataloader)
        image, gt, name = data['image'], data['gt'], data['name']
        image = image.to(device)
        gt = gt.to(device)
        output = seg_model(image)['out']
        output = F.interpolate(output, gt.shape[-2:], mode='bilinear', align_corners=False)
        loss += seg_loss(output, gt).item()
        pred = output.argmax(1)
        cm += fast_hist(pred, gt, num_classes)
    return {'loss': loss / set_size, 
            'pixel_accuracy': pixel_acc(cm).cpu().tolist(),
            'classwise_iou': per_class_iou(cm).cpu().tolist()}

def create_color_map(gt, labels2train, labels2palette):
    h, w = gt.shape[-2:]
    color_map = torch.ones((h, w, 3), dtype=torch.uint8, device=gt.device)
    for class_id in labels2train.keys():
        color_value = torch.tensor(labels2palette[class_id], dtype=torch.uint8, device=gt.device)
        color_map[gt == labels2train[class_id]] = color_value
    return color_map.permute(2, 0, 1)

def compile_summary_image(image, output, gt, mean, std, labels2train, labels2palette):
	std = torch.tensor(std, device=image.device).view(3, 1, 1)
	mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
	image = (image * std + mean) * 255
	image = image.type(torch.uint8)
	gt = create_color_map(gt, labels2train, labels2palette)
	pred = output.argmax(0)
	pred = create_color_map(pred, labels2train, labels2palette)
	grid = make_grid([image, pred, gt],
	                 nrow=1,
	                 normalize=False,
	                 scale_each=False)
	grid = F.interpolate(grid.type(torch.float).unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False)
	grid = grid.clamp(min=0, max=255).squeeze()
	return grid.type(torch.uint8)

def select_n_random(dataloader, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
