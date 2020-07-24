import os
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid, save_image


def fast_hist(pred, gt, num_classes):
    k = (gt >= 0) & (gt < num_classes)
    return torch.bincount(num_classes * pred[k].type(torch.int32) + gt[k], minlength=num_classes**2).view(num_classes, num_classes)

def per_class_iou(hist):
    return torch.diag(hist).float() / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).float()

def pixel_acc(hist):
    return torch.diag(hist).sum().float() / hist.sum().float()

def inf_iter(dataloader):
    while True:
        for batch_idx, data in enumerate(dataloader):
            yield batch_idx, data

def evaluate_segmentation_set(inf_dataloader, num_batches, seg_model, seg_loss, num_classes, device):
    loss = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    for i in range(num_batches):
        _, data = next(inf_dataloader)
        image, gt, name = data['image'], data['gt'], data['name']
        image = image.to(device)
        gt = gt.to(device)
        output = seg_model(image)['out']
        output = F.interpolate(output, gt.shape[-2:], mode='bilinear', align_corners=False)
        loss += seg_loss(output, gt).item()
        pred = output.argmax(1)
        cm += fast_hist(pred, gt, num_classes)
    return {'loss': loss / num_batches, 
            'pixel_accuracy': pixel_acc(cm).cpu().tolist(),
            'classwise_iou': per_class_iou(cm).cpu().tolist()}

def create_color_map(gt, labels2train, labels2palette):
    h, w = gt.shape[-2:]
    color_map = torch.ones((h, w, 3), dtype=torch.uint8, device=gt.device)
    for class_id in labels2train.keys():
        color_value = torch.tensor(labels2palette[class_id], dtype=torch.uint8, device=gt.device)
        color_map[gt == labels2train[class_id]] = color_value
    return color_map.permute(2, 0, 1)

def compile_summary_image(image, output, gt, mean, std, labels2train, labels2palette, scale_factor=0.5):
    std = torch.tensor(std, device=image.device).view(3, 1, 1)
    mean = torch.tensor(mean, device=image.device).view(3, 1, 1)
    image = (image * std + mean) * 255
    image = image.type(torch.uint8)
    gt_colored = create_color_map(gt, labels2train, labels2palette)
    pred = output.argmax(0)
    pred = create_color_map(pred, labels2train, labels2palette)
    grid = make_grid([image, pred, gt_colored],
                     nrow=1,
                     normalize=False,
                     scale_each=False)
    grid = F.interpolate(grid.type(torch.float).unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False)
    grid = grid.clamp(min=0, max=255).squeeze()
    return grid.type(torch.uint8)

def write_summary_images(writer, inf_dataloader, mean, std, labels2train, labels2palette, seg_model, device,
                         num_batches=1, tag='Images', scale_factor=0.5, global_step=0):
    for i in range(num_batches):
        _, data = next(inf_dataloader)
        image, gt, name = data['image'], data['gt'], data['name']
        image = image.to(device)
        gt = gt.to(device)
        output = seg_model(image)['out']
        output = F.interpolate(output, gt.shape[-2:], mode='bilinear', align_corners=False)
        for j in range(len(image)):
            summary_image = compile_summary_image(image[j], output[j], gt[j],
                mean, std, labels2train, labels2palette, scale_factor=scale_factor)
            writer.add_image(tag,
                img_tensor=summary_image,
                global_step=global_step+i+j,
                walltime=None,
                dataformats='CHW')

def write_predictions(inf_dataloader, seg_model, device, batches_to_visualize, predictions_path,
    mean, std, labels2train, labels2palette, prefix='src_'):
    std = torch.tensor(std, device=device).view(3, 1, 1)
    mean = torch.tensor(mean, device=device).view(3, 1, 1)
    for i in range(batches_to_visualize):
        _, data = next(inf_dataloader)
        image, gt, name = data['image'], data['gt'], data['name']
        image = image.to(device)
        gt = gt.to(device)
        output = seg_model(image)['out']
        output = F.interpolate(output, gt.shape[-2:], mode='bilinear', align_corners=False)
        pred = output.argmax(1)
        for j in range(len(image)):
            gt_colored = create_color_map(gt[j], labels2train, labels2palette)
            pred_colored = create_color_map(pred[j], labels2train, labels2palette)
            image_to_save = (image[j] * std + mean)
            image_to_save = image_to_save.clamp(min=0, max=1)
            pred_colored = pred_colored.clamp(min=0, max=255)
            gt_colored = gt_colored.clamp(min=0, max=255)
            save_image(image_to_save.type(torch.float),
                       os.path.join(predictions_path, prefix+'inp_'+name[j]))
            save_image(gt_colored.type(torch.float) / 255,
                       os.path.join(predictions_path, prefix+'gt_'+name[j]))
            save_image(pred_colored.type(torch.float) / 255,
                       os.path.join(predictions_path, prefix+'pred_'+name[j]))

def sample_features(inf_dataloader, seg_model, device, num_batches, pts_to_sample):
    deep_features = []
    labels = []
    for i in range(num_batches):
        _, data = next(inf_dataloader)
        image, gt, name = data['image'], data['gt'], data['name']
        image = image.to(device)
        gt = gt.to(device)
        features = seg_model.backbone(image)['out']
        gt_downscaled = F.interpolate(gt.unsqueeze(1).type(torch.float), features.shape[-2:], mode='nearest')
        gt_downscaled = gt_downscaled.type(gt.type())
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(-1, features.size(3))
        gt_downscaled = gt_downscaled.permute(0, 2, 3, 1)
        gt_downscaled = gt_downscaled.reshape(-1)
        perm = torch.randperm(features.size(0))
        idx = perm[:pts_to_sample]
        deep_features.append(features[idx].cpu())
        labels.append(gt_downscaled[idx].cpu())
    return torch.cat(deep_features, dim=0), torch.cat(labels, dim=0).tolist()

def create_embeddings(writer, inf_src_dataloader, inf_tar_dataloader, seg_model, device,
    num_batches, pts_to_sample):
    src_features, src_labels = sample_features(inf_src_dataloader, seg_model, device, num_batches, pts_to_sample)
    tar_features, tar_labels = sample_features(inf_tar_dataloader, seg_model, device, num_batches, pts_to_sample)
    dataset_label = ['src']*len(src_labels) + ['tar']*len(tar_labels)
    gt_label = src_labels + tar_labels
    all_features = torch.cat([src_features, tar_features], dim=0)
    all_labels = list(zip(gt_label, dataset_label))
    writer.add_embedding(all_features, metadata=all_labels, metadata_header=['gt', 'domain'])
