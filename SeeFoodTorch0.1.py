# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:06:03 2018

@author: MRVN
"""
from __future__ import print_function, division
import os
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from os.path import join
from PIL import Image
import numpy as np
import pickle













    
train_dir = "C:/Users/MRVN/Desktop/Kaggle/SeeFood/seefood/train/"    
test_dir = "C:/Users/MRVN/Desktop/Kaggle/SeeFood/seefood/test/"
hot_dog_image_dir = 'C:/Users/MRVN/Desktop/Kaggle/SeeFood/seefood/train/hot_dog'


not_hot_dog_image_dir = 'C:/Users/MRVN/Desktop/Kaggle/SeeFood/seefood/train/not_hot_dog'


images =  os.listdir(hot_dog_image_dir) + os.listdir(not_hot_dog_image_dir)
#import random
#random.shuffle(images)
#print(images)

classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
#print(classes)
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}
#print(class_to_idx)

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, Resize

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_size = (250, 250)
my_transform = Compose([Resize(image_size), ToTensor(), normalize])


traindata = ImageFolder(root=train_dir, transform=my_transform)
print(traindata[251])
trainloader = DataLoader(traindata, batch_size = 4, shuffle = True)
print(trainloader)


testdata = ImageFolder(root=test_dir, transform=my_transform)
testloader = DataLoader(testdata, batch_size = 4, shuffle = True)

print(iter(trainloader))
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)
# functions to show an image
"""

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

print(iter(trainloader))
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""

###############################################################################
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

import torchvision.models as models
resnet18 = models.resnet50(pretrained=False)
net = resnet18
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 2)

#criterion = nn.CrossEntropyLoss()


from torch.optim import lr_scheduler

import torch.optim as optim
# Observe that only parameters of final layer are being optimized as
# opoosed to before.
#optimizer_conv = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
net.avgpool = nn.AdaptiveAvgPool2d(1)
print(net)
device = "cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)

net.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
images, labels = images.to(device), labels.to(device)
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    

