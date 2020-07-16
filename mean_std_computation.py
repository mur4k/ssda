import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datasets.datasets import SegmentationDataSet


if __name__ == '__main__':
	print("Compiling the dataset")
	dataset = SegmentationDataSet(root='/nfs/students/mirlas/data', 
		image_list_name='cityscapes_images.txt', 
		label_list_name='cityscapes_labels.txt', 
		size=None, 
		mean=(0, 0, 0), 
		std=(1, 1, 1), 
		label2train=None)
	print("Compiling the dataloader")
	dataloader = DataLoader(dataset, batch_size=1)
	print("Length of the DataLoader:", len(dataloader))
	mean = 0.
	meansq = 0.
	std = 0.
	nb_samples = 0.
	for i, data in enumerate(dataloader):
		image = data['image']
		gt = data['gt']
		name = data['name']
		image = image.view(image.size(0), image.size(1), -1)
		mean += image.mean(2).sum(0)
		meansq += (image**2).mean(2).sum(0)
		nb_samples += image.size(0)
		if i == 0:
			print(image.shape)
			print(gt.shape)
			print(mean, meansq, nb_samples)
	print(nb_samples)
	mean = mean / nb_samples
	meansq = meansq / nb_samples
	std = torch.sqrt(meansq - mean**2)
	print('mean:', mean)
	print('std:', std)
