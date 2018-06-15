# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:38:44 2018

@author: MRVN
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os.path import join

hot_dog_image_dir = 'C:/Users/MRVN/Desktop/Kaggle/SeeFood/seefood/train/hot_dog'

hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = 'C:/Users/MRVN/Desktop/Kaggle/SeeFood/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

image_size = 224
img_paths = hot_dog_paths + not_hot_dog_paths

def show_pictures(image):
    """Show image with landmarks"""
    plt.imshow(image)

plt.figure()
show_pictures(io.imread(os.path.join(img_paths[0])))
plt.show()

class PictureDataset(Dataset):


    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(img_paths)

    def __getitem__(self, idx):
        image = io.imread(img_paths[idx])
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

picture_dataset = PictureDataset(root_dir=img_paths)
print("LÄÄÄÄÄÄÄÄÄÄÄÄÄÄNGE", len(picture_dataset))
fig = plt.figure()

for i in range(len(picture_dataset)):
    sample = picture_dataset[i]

    print(i, sample['image'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_pictures(**sample)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}


scale = Rescale((image_size,image_size))
#fig = plt.figure()
#sample = picture_dataset[3]
for i, tsfrm in enumerate([scale]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
#    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_pictures(**transformed_sample)

plt.show()


transformed_dataset = PictureDataset(root_dir=img_paths, transform = transforms.Compose([Rescale((image_size,image_size)), ToTensor()]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size())
print(len(transformed_dataset))

import torchvision.models as models
resnet50 = models.resnet50(pretrained=True)
test_data = transformed_dataset
preds = resnet50.predict(test_data)
print(preds)