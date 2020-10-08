import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils


class SegmentationDataSet(Dataset):
    def __init__(self, root, image_list_name, label_list_name, size, 
        mean=(0, 0, 0), std=(1, 1, 1), label2train=None):
        self.root = root
        self.image_list_name = image_list_name
        self.label_list_name = label_list_name
        # compile preprocessing transforms
        self.size = size
        self.label2train = label2train
        self.mean = mean
        self.std = std
        self.compose_transforms()
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

    def compose_transforms(self):
        self.image_preprocess_transform = transforms.Compose([
            transforms.Resize(self.size, Image.BILINEAR), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.label_preprocess_transform = transforms.Compose([
            transforms.Resize(self.size, Image.NEAREST),
            transforms.Lambda(lambda gt: torch.tensor(np.array(gt, dtype=np.int64)))])

    def read_data(self, index):
        image_path, label_path = self.files[index]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        name = os.path.basename(image_path)
        return image, label, name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, name = self.read_data(index)

        data = {'image': self.image_preprocess_transform(image), 
            'gt': self.label_preprocess_transform(label),
            'name': name}

        # re-assign labels to match the unified format
        if self.label2train is not None:
            for k, v in self.label2train.items():
                data['gt'][data['gt'] == k] = v

        return data


class RotationDataSet(SegmentationDataSet):
    def __init__(self, root, image_list_name, label_list_name, size, size_crop, 
        mean=(0, 0, 0), std=(1, 1, 1), label2train=None):
        self.rot_classes = 3
        self.size_crop = size_crop
        super().__init__(root, image_list_name, label_list_name, size, mean, std, label2train)

    def compose_transforms(self):
        self.image_preprocess_transform = transforms.Compose([
            transforms.Resize(self.size, Image.BILINEAR)])
        self.image_aux_preprocess_transform = transforms.Compose([
            transforms.RandomCrop(self.size_crop)])
        self.aux_transform = transforms.functional.rotate
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.label_preprocess_transform = transforms.Compose([
            transforms.Resize(self.size, Image.NEAREST),
            transforms.Lambda(lambda gt: torch.tensor(np.array(gt, dtype=np.int64)))])

    def __getitem__(self, index):
        image, label, name = self.read_data(index)

        image = self.image_preprocess_transform(image)
        aux_image = self.image_aux_preprocess_transform(image)

        aux_label = torch.randint(self.rot_classes+1, size=(1,)).squeeze()  # added 1 for class 0: unrotated
        aux_image = self.aux_transform(img=aux_image, angle=aux_label*90)

        data = {'image': self.normalize_transform(image), 
            'aux_image': self.normalize_transform(aux_image),
            'gt': self.label_preprocess_transform(label),
            'aux_gt': aux_label,
            'name': name}

        # re-assign labels to match the unified format
        if self.label2train is not None:
            for k, v in self.label2train.items():
                data['gt'][data['gt'] == k] = v

        return data
    
    
class JigsawDataSet(SegmentationDataSet):
    def __init__(self, root, image_list_name, label_list_name, size, grid_size, 
        mean=(0, 0, 0), std=(1, 1, 1), label2train=None):
        self.grid_size = grid_size
        self.tile_size = (size[0] // grid_size[0], size[1] // grid_size[1])
        super().__init__(root, image_list_name, label_list_name, size, mean, std, label2train)

    def jigsaw_transform(self, img):
        aux_label = torch.randperm(self.grid_size[0]*self.grid_size[1])
        tiles = []
        for i, j in enumerate(aux_label):
            top = int(j) // self.grid_size[1] * self.tile_size[0]
            left = int(j) % self.grid_size[1] * self.tile_size[1]
            height = self.tile_size[0]
            width = self.tile_size[1]
            tile = transforms.functional.crop(img, top, left, height, width)
            tiles.append(self.normalize_transform(tile))
        aux_image = utils.make_grid(tiles, nrow=self.grid_size[1], padding=0)
        return aux_label.view(self.grid_size), aux_image

    def compose_transforms(self):
        self.image_preprocess_transform = transforms.Compose([
            transforms.Resize(self.size, Image.BILINEAR)])
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.label_preprocess_transform = transforms.Compose([
            transforms.Resize(self.size, Image.NEAREST),
            transforms.Lambda(lambda gt: torch.tensor(np.array(gt, dtype=np.int64)))])

    def __getitem__(self, index):
        image, label, name = self.read_data(index)

        image = self.image_preprocess_transform(image)
        aux_label, aux_image = self.jigsaw_transform(image)

        data = {'image': self.normalize_transform(image), 
            'aux_image': aux_image,
            'gt': self.label_preprocess_transform(label),
            'aux_gt': aux_label,
            'name': name}

        # re-assign labels to match the unified format
        if self.label2train is not None:
            for k, v in self.label2train.items():
                data['gt'][data['gt'] == k] = v

        return data
