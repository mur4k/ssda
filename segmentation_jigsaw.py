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

from utils.utils import *
from losses.losses import *
from datasets.datasets import SegmentationDataSet, JigsawDataSet
from datasets.data_constants import *
from models.classifier import FCClassifier, FCClassifierBatchNorm

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
    snapshots_dir, log_dir, pred_dir, load_checkpoint,
    source_train_images, source_train_labels, 
    source_val_images, source_val_labels, 
    target_train_images, target_train_labels, 
    target_val_images, target_val_labels, 
    train_batch_size, val_batch_size,
    image_height, image_width, 
    grid_size, num_workers,
    backbone, classification_head, pretrained_backbone, 
    segmentation_loss, gamma,
    aux_injection_point, lambda_aux,
    learning_rate, momentum, weight_decay, 
    learning_rate_aux, betas_aux,
    max_iter, lrs_power,
    batches_to_eval_train, batches_to_visualize, 
    points_to_sample, save_step, display_step, seed):

    logging.info('Received the following configuration:')
    logging.info(f'Source_dir: {source_dir}, target_dir: {target_dir}, '
                 f'snapshots_dir: {snapshots_dir}, log_dir: {log_dir}, pred_dir: {pred_dir}, load_checkpoint: {load_checkpoint}, '
                 f'source_train_images: {source_train_images}, source_train_labels: {source_train_labels}, '
                 f'source_val_images: {source_val_images}, source_val_labels: {source_val_labels}, '
                 f'source_train_images: {target_train_images}, source_train_labels: {target_train_labels}, '
                 f'target_val_images: {target_val_images}, target_val_labels: {target_val_labels}, '
                 f'train_batch_size: {train_batch_size}, val_batch_size: {val_batch_size}, '
                 f'image_height: {image_height}, image_width: {image_width}, '
                 f'grid_size: {grid_size}, num_workers: {num_workers}, '
                 f'backbone: {backbone}, classification_head: {classification_head}, pretrained_backbone: {pretrained_backbone}, '
                 f'segmentation_loss: {segmentation_loss}, gamma: {gamma}, '
                 f'aux_injection_point: {aux_injection_point}, lambda_aux: {lambda_aux}, '
                 f'learning_rate: {learning_rate}, momentum: {momentum}, weight_decay: {weight_decay}, '
                 f'learning_rate_aux: {learning_rate_aux}, betas_aux: {betas_aux}, '
                 f'max_iter: {max_iter}, lrs_power: {lrs_power}, '
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
        'grid' + str(grid_size[0]) + 'x' + str(grid_size[1]), 
        ('IN' if pretrained_backbone else ''),
        segmentation_loss,
        aux_injection_point,
        'gamma{:.1e}_lmbdaux{:.1e}_lr{:.1e}_lraux{:.1e}_m{:.1e}_wd{:.1e}_lrsp{:.1e}'.format(
            gamma, lambda_aux, learning_rate, learning_rate_aux, momentum, weight_decay, lrs_power)
        ])
    start_epoch = 0
    if seed > 0:
        torch.manual_seed(seed)
    src_label = 1.
    tar_label = 0.

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
    target_train_dataset = JigsawDataSet(root=target_dir, image_list_name=target_train_images, 
                            label_list_name=target_train_labels, size=image_size, grid_size=grid_size,
                            mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD, label2train=CITYSCAPES_LABELS2TRAIN)
    target_val_dataset = SegmentationDataSet(root=target_dir, image_list_name=target_val_images, 
                            label_list_name=target_val_labels, size=image_size, 
                            mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD, label2train=CITYSCAPES_LABELS2TRAIN)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=train_batch_size,
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)
    target_train_dataloader = DataLoader(target_train_dataset, batch_size=train_batch_size,
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)
    source_val_dataloader = DataLoader(source_val_dataset, batch_size=val_batch_size, 
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)
    target_val_dataloader = DataLoader(target_val_dataset, batch_size=val_batch_size, 
                                shuffle=True, pin_memory=cuda_status, drop_last=True, num_workers=num_workers)

    inf_source_train_dataloader = inf_iter(source_train_dataloader)
    inf_target_train_dataloader = inf_iter(target_train_dataloader)
    inf_source_val_dataloader = inf_iter(source_val_dataloader)
    inf_target_val_dataloader = inf_iter(target_val_dataloader)

    max_epochs = max_iter // len(source_train_dataloader)

    #  initialize the model and the loss
    logging.info('Initializing models&losses')
    #  segmentation model
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
        raise ValueError("There are two losses currently supported: [ce, focal]. "
                         "Got: {}".format(segmentation_loss))
    #  rotation classifier
    if aux_injection_point == 'output':
        classifier_config = {
            'input_dim': NUM_CLASSES,
            'ndf': 64
        }
    elif aux_injection_point == 'feature':
        classifier_config = {
            'input_dim': 2048,
            'ndf': 256
        }
    else:
        raise ValueError("There are two injection points currently supported: [output, feature]. "
                         "Got: {}".format(aux_injection_point))
    classifier_config['num_classes'] = grid_size[0]*grid_size[1]
    classifier_config['avg_pool'] = True
    classifier_config['avg_pool_size'] = grid_size
    aux_model = FCClassifierBatchNorm(**classifier_config)
    aux_loss = torch.nn.CrossEntropyLoss()

    #  initialize the optimizers 
    #  we use different lrs for the classifier and the backbone if the latter one is pretrained
    logging.info('Initializing optimizers')
    optimizer = torch.optim.SGD([{'params': seg_model.backbone.parameters(), 
                                  'lr':learning_rate},
                                 {'params': seg_model.classifier.parameters(), 
                                  'lr':learning_rate * (10 if pretrained_backbone else 1)}],
                                lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer.zero_grad()
    aux_optimizer = torch.optim.Adam(aux_model.parameters(),
                            lr=learning_rate_aux, betas=betas_aux)
    aux_optimizer.zero_grad()

    #  initialize the learning rate scheduler
    logging.info('Initializing lr schedulers')
    lr_poly = lambda epoch: (1 - epoch / max_epochs) ** lrs_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_poly, last_epoch=-1)
    aux_scheduler = torch.optim.lr_scheduler.LambdaLR(aux_optimizer, lr_lambda=lr_poly, last_epoch=-1)

    #  reinitialize if nesseccary
    if os.path.isfile(load_checkpoint):
        logging.info('Loading the checkpoint')
        checkpoint = torch.load(load_checkpoint)
        if 'seg_model' in checkpoint:
            seg_model.load_state_dict(checkpoint['seg_model'])
        if 'aux_model' in checkpoint:
            aux_model.load_state_dict(checkpoint['aux_model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'aux_optimizer' in checkpoint:
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'aux_scheduler' in checkpoint:
            aux_scheduler.load_state_dict(checkpoint['aux_scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

    #  move everything to a device before training
    logging.info('Transfer models&optimizers to a device')
    transfer_model_and_optimizer(seg_model, optimizer, device)
    transfer_model_and_optimizer(aux_model, aux_optimizer, device)

    #  initialize the SummaryWriter
    logging.info('Initializing the SummaryWriter')
    summary_path = os.path.join(log_dir, model_name)
    writer = SummaryWriter(log_dir=summary_path)

    #  initialize loss, accuracy values
    running_src_seg_loss = 0.0
    running_tar_aux_loss = 0.0
    running_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)

    #  train the model
    logging.info('Start training')
    seg_model.train()
    aux_model.train()
    epoch = 0
    batch_idx = 0
    for num_iter in range(start_epoch*len(source_train_dataloader), 
                          max_epochs*len(source_train_dataloader)):
        if num_iter % len(source_train_dataloader) == 0:
            epoch = num_iter // len(source_train_dataloader)
            logging.info(f'Epoch: {epoch}/{max_epochs}')
        batch_idx = num_iter % len(source_train_dataloader)
        
        #  zero the parameter gradients
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
            
        #  forward pass TARGET image
        #  load target data
        _, data = next(inf_target_train_dataloader)
        aux_image, aux_gt, name = data['aux_image'], data['aux_gt'], data['name']
        aux_image = aux_image.to(device)
        aux_gt = aux_gt.to(device)

        #  forward pass encoder + segmentation
        features = seg_model.backbone(aux_image)
        output = seg_model.classifier(features['out'])
        output = F.interpolate(output, aux_image.shape[-2:], 
                    mode='bilinear', align_corners=False)

        if aux_injection_point == 'output':
            aux_output = aux_model(output)
        else:
            aux_output = aux_model(features['out'])

        #  backprop aux_model
        tar_aux_loss = lambda_aux * aux_loss(aux_output, aux_gt)
        tar_aux_loss.backward()

        #  forward pass SOURCE image
        #  load source data
        _, data = next(inf_source_train_dataloader)
        image, gt, name = data['image'], data['gt'], data['name']
        image = image.to(device)
        gt = gt.to(device)

        #  forward pass encoder + segmentation
        features = seg_model.backbone(image)
        output = seg_model.classifier(features['out'])
        output = F.interpolate(output, image_size, 
                    mode='bilinear', align_corners=False)

        #  backprop encoder + segmentation
        src_seg_loss = seg_loss(output, gt)
        src_seg_loss.backward()

        optimizer.step()
        aux_optimizer.step()
        
        #  update metrics
        with torch.no_grad():
            running_src_seg_loss += src_seg_loss.item()
            running_tar_aux_loss += tar_aux_loss.item()
            src_seg_pred = output.argmax(1)
            running_cm += fast_hist(src_seg_pred, gt, NUM_CLASSES)

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

            logging.info('Iteration {}/{}: Loss: {:.4f} | Pixel accuracy: {:.4f}'.format(batch_idx+1, 
                                                                                         len(source_train_dataloader), 
                                                                                         src_seg_loss.item(), 
                                                                                         pixel_train_acc))

            #  write summary
            writer.add_scalars('Seg_loss', 
                {'training_src_seg_loss': running_src_seg_loss/display_step,
                 'validation_src_seg_loss': src_val_metrics['loss'],
                 'validation_tar_seg_loss': tar_val_metrics['loss']},
                num_iter)
            writer.add_scalars('Aux_loss', 
                {'training_tar_fake_loss': running_tar_aux_loss/lambda_aux/display_step},
                num_iter)
            writer.add_scalars('Pixel_iou',
                {'training_src_pixel_acc': pixel_train_acc,
                 'validation_src_pixel_acc': src_val_metrics['pixel_accuracy'],
                 'validation_target_pixel_acc': tar_val_metrics['pixel_accuracy']},
                num_iter)
            for i in range(NUM_CLASSES):
                writer.add_scalar('Class_iou/{}'.format(CLASSES[i]),
                    classwise_train_iou[i] if not classwise_train_iou[i] == float('nan') else 0.0,
                    num_iter)

            #  reset loss, accuracy values
            running_src_seg_loss = 0.0
            running_tar_aux_loss = 0.0
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
                    train_src_summary_image = compile_summary_image(image[i], output[i], gt[i], 
                        GTA5_MEAN, GTA5_STD, GTA5_LABELS2TRAIN, GTA5_LABELS2PALETTE, scale_factor=0.5)
                    writer.add_image('Train/Src',
                        img_tensor=train_src_summary_image, 
                        global_step=num_iter+i, 
                        walltime=None, 
                        dataformats='CHW')
                #  val
                write_summary_images(writer, inf_source_val_dataloader, 
                    GTA5_MEAN, GTA5_STD, GTA5_LABELS2TRAIN, GTA5_LABELS2PALETTE, 
                    seg_model, device, 
                    num_batches=1, tag='Val/Src', scale_factor=0.5, global_step=num_iter)
                write_summary_images(writer, inf_target_val_dataloader, 
                    CITYSCAPES_MEAN, CITYSCAPES_STD, CITYSCAPES_LABELS2TRAIN, CITYSCAPES_LABELS2PALETTE, 
                    seg_model, device, 
                    num_batches=1, tag='Val/Tar', scale_factor=0.5, global_step=num_iter)
        
        #  end_of_epoch
        if (num_iter + 1) % len(source_train_dataloader) == 0:
            #  decay lr
            logging.info('LR update')
            scheduler.step()
            aux_scheduler.step()

            #  save the model    
            if (epoch + 1) % save_step == 0:
                logging.info('Save the model')
                save_path = os.path.join(snapshots_dir, model_name, 'checkpoint_{}.pth'.format(epoch+1))
                torch.save({
                    'seg_model': seg_model.state_dict(),
                    'aux_model': aux_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'aux_optimizer': aux_optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'aux_scheduler': aux_scheduler.state_dict(),
                    'epoch': epoch}, save_path)
        
    #  evaluate the model
    seg_model.eval()
    aux_model.eval()
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
