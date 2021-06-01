#! /usr/bin/env python

# PyTorch Tutorial 14 - Convolutional Neural Network (CNN)
# https://www.youtube.com/watch?v=pDdP0TFzsoQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=14

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - CNN Theory overview
# 1m - concepts CNN convolutional neural net
# 0m12 - CIFAR-10 dataset - https://en.wikipedia.org/wiki/CIFAR-10
# 4m - Code start, GPU support, hyper-parameters
# 4m40 - load CIFAR dataset
# 5m - Quick walk the code see code structure
# 7m - class definitions in detail
# 7m23 - CNN architecture slide
# 11m - going over chosen layer parameters
# 13m46 - Calculating the output size > Inputs into Linear layers
# 17m20 - Class forward method layers
# 20m30 - Run training
#
#
#
#

# - - - Code Structure - - -
# GPU support
# hyper-parameters
# load (CIFAR) dataset
# class definitions for model
# instantiate model
# assign cirterion
# assign optimizer
# train model
# test model

# CNN architecture slide
#

# Classification O/P
#  |
# Softmax layer spread output into a proportional representation
#  |
# Features Flattened into 1d fully connected layer connects to 2 more layers?
#  |
# Pooling: (Downsampling, stops overfitting)
# Convolution & ReLU
#  |
# Pooling: (Downsampling, stops overfitting)
# Convolution & ReLU
#  |
# INPUT
#
# NOTE
# CONCVOLUTION & ReLU is done on co-located areas to preserve spacial information
# POOLING down samples - removes resolution to stop overfitting
# these layers are repeated feeding forward into the next layer - FOR FEATURE EXTRACTION


import torch

import torch.nn as nn       # PyTorch nn module has high-level APIs to build a neural network.
  # Torch. nn module uses Tensors and Automatic differentiation modules for training and building layers such as input,
  # hidden, and output layers - DOCS: https://pytorch.org/docs/stable/nn.html

import torch.nn.functional as F # https://pytorch.org/docs/stable/nn.functional.html
  # What is the difference between torch.nn and torch.nn.functional?            FROM discuss.pytorch.org
  #
  # While the former defines nn.Module classes, the latter uses a functional (stateless) approach.
  #
  # - - torch.nn - Module classes
  # nn.Modules are defined as Python classes and have attributes, e.g. a nn.Conv2d module will
  # have some internal attributes like self.weight.
  #
  # - - torch.nn.functional - function versions
  # F.conv2d however just defines the operation and needs all arguments
  # to be passed (including the weights and bias).

  # Internally the modules will usually call their functional counterpart in the forward method somewhere.


import torchvision
  # popular datasets, model architectures, and common image transformations for computer vision.
  # conda install torchvision -c pytorch    # already installed.

import torchvision.transforms as transforms
  # transforms that can be applied to Tensors - can be 'composed' into sets of transforms
  # https://pytorch.org/vision/stable/transforms.html
  # see 10_tensor_dataset_transforms.py &

import numpy as np      # NumPy is a library for the Python programming language, adding support for large,
  # multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate
  # on these arrays - DOCS: https://numpy.org/doc/stable/user/whatisnumpy.html

import matplotlib.pyplot as plt # Matplotlib is a plotting library for the Python programming language. It provides an
  # object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter,
  # wxPython, Qt, or GTK - DOCS:
  # cheatsheets: https://github.com/matplotlib/cheatsheets#cheatsheets
  # How to plot & save graph hello world: https://github.com/UnacceptableBehaviour/latex_maths#python---matplotlib-numpy


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# TODO make 12x8 set of images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))



# 7m - class definitions in detail
# 7m23 - CNN architecture slide
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Conv2d     https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        # MaxPool2d  https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        # Linear     https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels = 3 (RGB), out_channels = 6?, kernel_size = 5 (5x5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
