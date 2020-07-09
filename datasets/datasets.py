import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SegmentationDataSet(Dataset):
    def __init__(self, root, image_list_name, label_list_name, size=None, mean=(0, 0, 0), std=(1, 1, 1), label2train=None):
        self.root = root
        self.image_list_name = image_list_name
        self.label_list_name = label_list_name
        self.size = size
        self.label2train = label2train
        self.mean = mean
        self.std = std
        # read lists of files
        self.image_list_path = os.path.join(root, self.image_list_name)
        self.label_list_path = os.path.join(root, self.label_list_name)
        with open(self.image_list_path) as images_list:
    		self.images_paths = [os.path.join(self.root, p.strip()) for p in images_list]
    	self.images_paths = [p for p in self.images_paths if os.path.isfile(p)]
    	with open(self.label_list_path) as labels_list:
    		self.labels_paths = [os.path.join(self.root, p.strip()) for p in label_list]
    	self.labels_paths = [p for p in self.labels_paths if os.path.isfile(p)]
    	self.files = list(zip(self.images_paths, self.labels_paths))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        image_path, label_path = self.files[index]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        name = os.path.basename(image_path)

        # resize
        if self.size is not None:
        	image = image.resize(self.size, Image.BICUBIC)
        	label = label.resize(self.size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.int8)

        # re-assign labels to match the unified format
        if self.label2train is None:
        	gt = label.copy()
        else:
        	gt = 255 * np.ones(label.shape, dtype=np.int8)
        	for k, v in self.label2train.items():
            	gt[label == k] = v

        size = image.shape
        image -= self.mean
        image /= self.std
        image = image.transpose((2, 0, 1))

        return {'image': transforms.ToTensor()(image), 
        	    'gt': transforms.ToTensor()(gt),
        	    'name': name}


if __name__ == '__main__':
	dataset = SegmentationDataSet(root='/nfs/students/mirlas/data', image_list_name='gta5_images.txt', label_list_name='gta5_labels.txt', size=None, mean=(0, 0, 0), std=(1, 1, 1), labels2train=None)
	dataloader = data.DataLoader(dataset, batch_size=10)
	print("Length of the DataLoader:", len(dataloader))
	mean = 0.
	std = 0.
	nb_samples = 0.
	for i, data in enumerate(dataloader)
		image = data['image']
		gt = data['gt']
		name = data['name']
		if i == 0:
			print(image.shape)
			print(gt.shape)
		image = image.view(image.size(0), image.size(1), -1)
		mean += image.mean(2).sum(0)
    	meansq += (image**2).mean(2).sum(0)
    	nb_samples += image.size(0)
    std = torch.sqrt((meansq - mean**2) / nb_samples)
    mean = mean / nb_samples
    print('mean:', mean)
    print('std:', std)