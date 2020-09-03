import logging
import os
from sacred import Experiment
import numpy as np
import seml

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from losses.losses import *
from utils.utils import *
from datasets.datasets import SegmentationDataSet
from datasets.data_constants import *

ex = Experiment()
seml.experiment.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.observers.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(source_dir, target_dir, 
    snapshots_dir, log_dir, pred_dir,
    source_train_images, source_train_labels, 
    source_val_images, source_val_labels, 
    target_val_images, target_val_labels, 
    train_batch_size, val_batch_size,
    image_height, image_width, num_workers, 
    backbone, classification_head, pretrained_backbone, 
    segmentation_loss, gamma,
    learning_rate, momentum, weight_decay,
    max_iter, epoch_to_resume, lrs_power,
    batches_to_eval_train, batches_to_visualize, 
    points_to_sample, save_step, display_step, seed):

    logging.info('Received the following configuration:')
    logging.info(f'Source_dir: {source_dir}, target_dir: {target_dir}, '
                 f'snapshots_dir: {snapshots_dir}, log_dir: {log_dir}, pred_dir: {pred_dir}, '
                 f'source_train_images: {source_train_images}, source_train_labels: {source_train_labels}, '
                 f'source_val_images: {source_val_images}, source_val_labels: {source_val_labels}, '
                 f'target_val_images: {target_val_images}, target_val_labels: {target_val_labels}, '
                 f'train_batch_size: {train_batch_size}, val_batch_size: {val_batch_size}, '
                 f'image_height: {image_height}, image_width: {image_width}, num_workers: {num_workers}, '
                 f'backbone: {backbone}, classification_head: {classification_head}, pretrained_backbone: {pretrained_backbone}, '
                 f'segmentation_loss: {segmentation_loss}, gamma: {gamma}, '
                 f'learning_rate: {learning_rate}, momentum: {momentum}, weight_decay: {weight_decay}, '
                 f'max_iter: {max_iter}, epoch_to_resume: {epoch_to_resume}, lrs_power: {lrs_power}, '
                 f'batches_to_eval_train: {batches_to_eval_train}, batches_to_visualize: {batches_to_visualize}, '
                 f'points_to_sample: {points_to_sample}, save_step:{save_step}, display_step: {display_step}, seed: {seed}')
    
    #  initialize the global parameters
    cuda_status = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_status else "cpu")
    image_size = (image_height, image_width)
    model_name = '_'.join(
        [backbone,
        classification_head,
        str(image_height) + 'x' + str(image_width),
        ('IN' if pretrained_backbone else ''),
        segmentation_loss,
        'gamma{:.1e}_lr{:.1e}_m{:.1e}_wd{:.1e}_lrsp{:.1e}'.format(
            gamma, learning_rate, momentum, weight_decay, lrs_power)
        ])
    start_epoch = 0
    if seed > 0:
        torch.manual_seed(seed)

    #  create neccessary directories for snapshots and logs
    if not os.path.exists(os.path.join(snapshots_dir, model_name)):
        os.makedirs(os.path.join(snapshots_dir, model_name))
    if not os.path.exists(os.path.join(log_dir, model_name)):
        os.makedirs(os.path.join(log_dir, model_name))
    if not os.path.exists(os.path.join(pred_dir, model_name)):
        os.makedirs(os.path.join(pred_dir, model_name))
    
    #  initialize the datasets
    logging.info('Initializing the datasets')
    source_train_dataset = SegmentationDataSet(root=source_dir, image_list_name=source_train_images, 
                            label_list_name=source_train_labels, size=image_size, 
                            mean=GTA5_MEAN, std=GTA5_STD, label2train=GTA5_LABELS2TRAIN)
    source_val_dataset = SegmentationDataSet(root=target_dir, image_list_name=source_val_images, 
                            label_list_name=source_val_labels, size=image_size, 
                            mean=GTA5_MEAN, std=GTA5_STD, label2train=GTA5_LABELS2TRAIN)
    target_val_dataset = SegmentationDataSet(root=target_dir, image_list_name=target_val_images, 
                            label_list_name=target_val_labels, size=image_size, 
                            mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD, label2train=CITYSCAPES_LABELS2TRAIN)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=train_batch_size,
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)
    source_val_dataloader = DataLoader(source_val_dataset, batch_size=val_batch_size, 
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=val_batch_size, 
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)

    inf_source_val_dataloader = inf_iter(source_val_dataloader)
    inf_target_val_dataloader = inf_iter(target_val_dataloader)

    max_epochs = max_iter // len(source_train_dataloader)

    #  initialize the model and the loss
    logging.info('Initializing models&losses')
    seg_model_name =  classification_head + '_' + backbone
    seg_model_loader = getattr(models.segmentation, seg_model_name)
    seg_model = seg_model_loader(pretrained=False, 
                    num_classes=NUM_CLASSES,
                    progress=False,
                    aux_loss=False,
                    pretrained_backbone=pretrained_backbone)
    if segmentation_loss == 'ce':
        seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    elif segmentation_loss == 'focal':
        seg_loss = FocalLoss(alpha=1.0, 
                    gamma=gamma, 
                    ignore_index=255,
                    reduction='mean')
    else:
        raise ValueError("There are only two losses currently supported: [ce, focal]. Got: {}".format(segmentation_loss))

    #  initialize the optimizer 
    #  we use different lrs for the classifier and the backbone if the latter one is pretrained
    logging.info('Initializing optimizers')
    optimizer = torch.optim.SGD([{'params': seg_model.backbone.parameters(), 
                                  'lr':learning_rate},
                                 {'params': seg_model.classifier.parameters(), 
                                  'lr':learning_rate * (10 if pretrained_backbone else 1)}],
                                lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer.zero_grad()

    #  initialize the learning rate scheduler
    logging.info('Initializing lr schedulers')
    lr_poly = lambda epoch: (1 - epoch / max_epochs) ** lrs_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_poly, last_epoch=-1)

    #  reinitialize if nesseccary
    checkpoint_path = os.path.join(snapshots_dir, model_name, 'checkpoint_{}.pth'.format(epoch_to_resume))
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'seg_model' in checkpoint:
            seg_model.load_state_dict(checkpoint['seg_model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'loss' in checkpoint:
            loss = checkpoint['loss']

    #  move everything to a device before training
    logging.info('Transfer models&optimizer to a device')
    transfer_model_and_optimizer(seg_model, optimizer, device)

    #  initialize the SummaryWriter
    logging.info('Initializing the SummaryWriter')
    summary_path = os.path.join(log_dir, model_name)
    writer = SummaryWriter(log_dir=summary_path)

    #  initialize loss, accuracy values
    running_loss = 0.0
    running_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    #  train the model
    logging.info('Start training')
    seg_model.train()
    for epoch in range(start_epoch, max_epochs):
        logging.info(f'Epoch: {epoch}/{max_epochs}')
        #  iterate over the dataset
        for batch_idx, data in enumerate(source_train_dataloader):
            #  zero the parameter gradients
            optimizer.zero_grad()

            #  load data
            image, gt, name = data['image'], data['gt'], data['name']
            image = image.to(device)
            gt = gt.to(device)

            #  forward pass
            features = seg_model.backbone(image)
            output = seg_model.classifier(features['out'])
            output = F.interpolate(output, image_size, mode='bilinear', align_corners=False)

            #  backprop
            loss = seg_loss(output, gt)
            loss.backward()
            optimizer.step()

            #  update metrics
            with torch.no_grad():
                running_loss += loss.item()
                pred = output.argmax(1)
                running_cm += fast_hist(pred, gt, NUM_CLASSES)

            #  update the training summary stats: metrics and images
            if (batch_idx + 1) % display_step == 0:
                with torch.no_grad():
                    #  train
                    pixel_train_acc = pixel_acc(running_cm)
                    classwise_train_iou = per_class_iou(running_cm)

                    #  val
                    src_val_metrics = evaluate_segmentation_set(inf_source_val_dataloader, batches_to_eval_train, 
                        seg_model, seg_loss, NUM_CLASSES, device)
                    tar_val_metrics = evaluate_segmentation_set(inf_target_val_dataloader, batches_to_eval_train, 
                        seg_model, seg_loss, NUM_CLASSES, device)
                
                logging.info('Iteration {}/{}: Loss: {:.4f} | Pixel accuracy: {:.4f}'.format(batch_idx+1, len(source_train_dataloader), loss.item(), pixel_train_acc))

                #  write summary
                writer.add_scalars('Loss', 
                    {'training_src_loss': running_loss/display_step,
                     'validation_src_loss': src_val_metrics['loss'],
                     'validation_target_loss': tar_val_metrics['loss']},
                    epoch*len(source_train_dataloader)+batch_idx)
                writer.add_scalars('Pixel_iou',
                    {'training_src_pixel_acc': pixel_train_acc,
                     'validation_src_pixel_acc': src_val_metrics['pixel_accuracy'],
                     'validation_target_pixel_acc': tar_val_metrics['pixel_accuracy']},
                    epoch*len(source_train_dataloader)+batch_idx)
                for i in range(NUM_CLASSES):
                    writer.add_scalar('Class_iou/{}'.format(CLASSES[i]),
                        classwise_train_iou[i] if not classwise_train_iou[i] == float('nan') else 0.0,
                        epoch*len(source_train_dataloader)+batch_idx)

                #  reset loss, accuracy values
                running_loss = 0.0
                running_cm.fill_(0.0)

            #  write images to the summary (4 times per epoch)
            if (batch_idx + 1) == len(source_train_dataloader) or \
                (batch_idx + 1) == len(source_train_dataloader) // 2 or \
                (batch_idx + 1) == len(source_train_dataloader) // 4 or \
                (batch_idx + 1) == 3 * len(source_train_dataloader) // 4: 
                logging.info('Writing summary images')
                with torch.no_grad():
                    #  train
                    for i in range(len(image)):
                        train_summary_image = compile_summary_image(image[i], output[i], gt[i], 
                            GTA5_MEAN, GTA5_STD, GTA5_LABELS2TRAIN, GTA5_LABELS2PALETTE, scale_factor=0.5)
                        writer.add_image('Train/Src',
                            img_tensor=train_summary_image, 
                            global_step=epoch*len(source_train_dataloader)+batch_idx+i, 
                            walltime=None, 
                            dataformats='CHW')
                    #  val
                    write_summary_images(writer, inf_source_val_dataloader, 
                        GTA5_MEAN, GTA5_STD, GTA5_LABELS2TRAIN, GTA5_LABELS2PALETTE, 
                        seg_model, device, 
                        num_batches=1, tag='Val/Src', scale_factor=0.5, global_step=epoch*len(source_train_dataloader)+batch_idx)
                    write_summary_images(writer, inf_target_val_dataloader, 
                        CITYSCAPES_MEAN, CITYSCAPES_STD, CITYSCAPES_LABELS2TRAIN, CITYSCAPES_LABELS2PALETTE, 
                        seg_model, device, 
                        num_batches=1, tag='Val/Tar', scale_factor=0.5, global_step=epoch*len(source_train_dataloader)+batch_idx)

        #  decay lr
        scheduler.step()

        #  save the model
        if (epoch + 1) % save_step == 0:
            save_path = os.path.join(snapshots_dir, model_name, 'checkpoint_{}.pth'.format(epoch+1))
            torch.save({
                'seg_model': seg_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}, save_path)
        
    #  evaluate the model
    seg_model.eval()
    with torch.no_grad():
        logging.info('Evaluate src')
        src_val_metrics = evaluate_segmentation_set(inf_source_val_dataloader, len(source_val_dataloader), 
                            seg_model, seg_loss, NUM_CLASSES, device)
        logging.info('Evaluate tar')
        tar_val_metrics = evaluate_segmentation_set(inf_target_val_dataloader, len(target_val_dataloader), 
                            seg_model, seg_loss, NUM_CLASSES, device)
    
    #  write results
    results = dict()
    results.update({'src_'+k: v for k, v in src_val_metrics.items()})
    results.update({'tar_'+k: v for k, v in tar_val_metrics.items()})

    #  create visualizations: t-SNE and qualititative predictions
    predictions_path = os.path.join(pred_dir, model_name)
    with torch.no_grad():
        logging.info('Write predcitions for src')
        write_predictions(source_val_dataloader, 
            seg_model, device, 
            batches_to_visualize, predictions_path, 
            GTA5_MEAN, GTA5_STD, GTA5_LABELS2TRAIN, 
            GTA5_LABELS2PALETTE, prefix='src_')
        logging.info('Write predcitions for tar')
        write_predictions(target_val_dataloader, 
            seg_model, device, 
            batches_to_visualize, predictions_path,
            CITYSCAPES_MEAN, CITYSCAPES_STD, CITYSCAPES_LABELS2TRAIN,
            CITYSCAPES_LABELS2PALETTE, prefix='tar_')
        logging.info('Create t-SNE embeddings')
        create_embeddings(writer, inf_source_val_dataloader, inf_target_val_dataloader,
            seg_model, device, 
            batches_to_visualize, points_to_sample)
    
    # the returned result will be written into the database
    return results
