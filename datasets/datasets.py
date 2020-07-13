import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SegmentationDataSet(Dataset):
	def __init__(self, root, image_list_name, label_list_name, size, 
		mean=(0, 0, 0), std=(1, 1, 1), label2train=None):
		self.root = root
		self.image_list_name = image_list_name
		self.label_list_name = label_list_name
		self.size = size
		self.label2train = label2train
		self.mean = mean
		self.std = std
		self.image_preprocess_transform = transforms.Compose([
			transforms.Resize(self.size, Image.BILINEAR), 
			transforms.ToTensor(), 
			transforms.Normalize(mean=self.mean, std=self.std)])
		self.label_preprocess_transform = transforms.Compose([
			transforms.Resize(self.size, Image.NEAREST),
			transforms.Lambda(lambda gt: torch.tensor(np.array(gt)))])
		# read lists of files
		self.image_list_path = os.path.join(root, self.image_list_name)
		self.label_list_path = os.path.join(root, self.label_list_name)
		with open(self.image_list_path) as images_list:
			self.images_paths = [os.path.join(self.root, p.strip()) for p in images_list]
		self.images_paths = [p for p in self.images_paths if os.path.isfile(p)]
		with open(self.label_list_path) as labels_list:
			self.labels_paths = [os.path.join(self.root, p.strip()) for p in labels_list]
		self.labels_paths = [p for p in self.labels_paths if os.path.isfile(p)]
		self.files = list(zip(self.images_paths, self.labels_paths))


	def __len__(self):
	    return len(self.files)


	def __getitem__(self, index):
		image_path, label_path = self.files[index]

		image = Image.open(image_path).convert('RGB')
		label = Image.open(label_path)
		name = os.path.basename(image_path)

		data = {'image': self.image_preprocess_transform(image), 
			'gt': self.label_preprocess_transform(label),
			'name': name}

		# re-assign labels to match the unified format
		if self.label2train is not None:
			for k, v in self.label2train.items():
				data['gt'][data['gt'] == k] = v
		
		return data
