#! /usr/bin/env python

# PyTorch Tutorial 14 - Convolutional Neural Network (CNN)
# https://www.youtube.com/watch?v=pDdP0TFzsoQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=14

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)
import sys

# 0m - CNN Theory overview
# 1m - concepts CNN convolutional neural net
# 0m12 - CIFAR-10 dataset - https://en.wikipedia.org/wiki/CIFAR-10
# 4m - Code start, GPU support, hyper-parameters
# 4m40 - load CIFAR dataset
# 5m - Quick walk the code see code structure
# 7m - class definitions in detail
# 7m23 - CNN architecture slide
# 11m - going over chosen layer parameters
# 13m - parameter explanation
# 13m46 - Conv2d o/p - Calculating the output size > Inputs into Linear layers
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
batch_size = 12
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

# shows a batch of images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))



# 7m - class definitions in detail
# 7m23 - CNN architecture slide
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Conv2d     https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        #
        # KERNEL = FILTER 3x3, 4x4 etc
        # FILTERS are convolved with image by scanning it: produces a FEATURE MAP
        # scanning is done by moving the filter x pixels at a time x is set by the STRIDE hyperparameter
        #
        # torch.nn.Conv2d(in_channels,  # inputs signal planes - for initial input of image this will usually be 3 for RGB planes
        #                 out_channels, # there will be one plane (feature map) for each filter
        #                 kernel_size,  # filter dimension:         3 3x3, 4x4 etc or (2,4) (height,width)
        #                 stride=1,     # filter movement amount    3 3x3, 4x4 etc or (2,4) (height,width)
        #                 padding=0,    # periferal padding         3 3x3, 4x4 etc or (2,4) (height,width)
        #                 dilation=1,   # separation between filter sample points  'a trous algorithm'  ^ as above int / tuple
        #                 groups=1,     # last piece of puzzle < need better understanding of this TODO
        #                 bias=True,    # learnable bias default = True
        #                 padding_mode='zeros')
        #
        # dilation stride anims: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        # maths 0 to hero: https://arxiv.org/pdf/1603.07285.pdf   https://www.youtube.com/watch?v=MmG2ah5Df4g
        #
        # MaxPool2d  https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        #
        # OUTPUT SIZE (Hout,Wout) = (Hin/stride, Win/stride)    See above link foe EQN tha includes all params
        #
        # torch.nn.MaxPool2d(kernel_size,           # pool window dimension:    3 3x3, 4x4 etc or (2,4) (height,width)
        #                    stride=None,           # uses kernel_size as default
        #                    padding=0,
        #                    dilation=1,
        #                    return_indices=False,
        #                    ceil_mode=False)
        #
        # Linear     https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        # fc - fully connected / dense layer
        #
        # torch.nn.Linear(in_features,      #
        #                 out_features,
        #                 bias=True)

        # parameter explanation 13m
        self.conv1 = nn.Conv2d(3, 12, 3)         # in_channels = 3 (RGB), out_channels = 6 features / planes, kernel_size = 5 (5x5)
        self.pool = nn.MaxPool2d(2, 2)          # kernel = 2, stride=2  down sample  OP =
        self.conv2 = nn.Conv2d(12, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     # -> n, 3, 32, 32
    #     x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
    #     x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
    #     x = x.view(-1, 16 * 5 * 5)            # -> n, 400
    #     x = F.relu(self.fc1(x))               # -> n, 120
    #     x = F.relu(self.fc2(x))               # -> n, 84
    #     x = self.fc3(x)                       # -> n, 10
    #     return x

    def forward(self, x): # Conv2d o/p - Calculating the output size: (Wfmap = (Wimg - Wkrn + 2*padding) / stride ) + 1 - - -\
        print(f"\n shape( x ) \n{ x.shape }")                           # torch.Size([4, 3, 32, 32])                          |
                                                # -> n, 3, 32, 32                       batch size                            |
        act = F.relu(self.conv1(x))                                     #              /                                      |
        print(f"\n shape( F.relu(self.conv1(x)) ) \n{ act.shape }")     # torch.Size([4, 6, 28, 28])  < why this changes! <--/  from 32x32 to 28x28
        x = self.pool(act)                      # -> n, 6, 14, 14
        print(f"\n shape( self.pool(act) ) \n{ x.shape }")              # torch.Size([4, 6, 14, 14])
        act = F.relu(self.conv2(x))
        print(f"\n shape( F.relu(self.conv1(x)) ) \n{ act.shape }")     # torch.Size([4, 16, 10, 10])
        x = self.pool(act)                      # -> n, 16, 5, 5
        print(f"\n shape( self.pool(act) ) \n{ x.shape }")              # torch.Size([4, 16, 5, 5])

        # flatten for linear layers
        x = x.view(-1, 32 * 6 * 6)            # -> n, 400

        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10

        sys.exit(0)
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size (kernel = feature map)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './data/cnn14.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

# RESULT
# Epoch [1/5], Step [2000/12500], Loss: 2.2722
# Epoch [1/5], Step [4000/12500], Loss: 2.2892
# Epoch [1/5], Step [6000/12500], Loss: 2.2897
# Epoch [1/5], Step [8000/12500], Loss: 2.3078
# Epoch [1/5], Step [10000/12500], Loss: 2.1805
# Epoch [1/5], Step [12000/12500], Loss: 2.1557
# Epoch [2/5], Step [2000/12500], Loss: 1.6770
# Epoch [2/5], Step [4000/12500], Loss: 2.2442
# Epoch [2/5], Step [6000/12500], Loss: 1.8525
# Epoch [2/5], Step [8000/12500], Loss: 2.4297
# Epoch [2/5], Step [10000/12500], Loss: 1.7102
# Epoch [2/5], Step [12000/12500], Loss: 1.9221
# Epoch [3/5], Step [2000/12500], Loss: 1.7978
# Epoch [3/5], Step [4000/12500], Loss: 1.5310
# Epoch [3/5], Step [6000/12500], Loss: 1.8889
# Epoch [3/5], Step [8000/12500], Loss: 2.2381
# Epoch [3/5], Step [10000/12500], Loss: 1.3370
# Epoch [3/5], Step [12000/12500], Loss: 1.0403
# Epoch [4/5], Step [2000/12500], Loss: 1.3470
# Epoch [4/5], Step [4000/12500], Loss: 1.6430
# Epoch [4/5], Step [6000/12500], Loss: 1.1701
# Epoch [4/5], Step [8000/12500], Loss: 1.5667
# Epoch [4/5], Step [10000/12500], Loss: 1.8386
# Epoch [4/5], Step [12000/12500], Loss: 1.5948
# Epoch [5/5], Step [2000/12500], Loss: 1.6294
# Epoch [5/5], Step [4000/12500], Loss: 0.9172
# Epoch [5/5], Step [6000/12500], Loss: 1.3630
# Epoch [5/5], Step [8000/12500], Loss: 1.6863
# Epoch [5/5], Step [10000/12500], Loss: 1.2374
# Epoch [5/5], Step [12000/12500], Loss: 0.8387
# Finished Training - batch = 4
# Accuracy of the network: 48.71 %
# Accuracy of plane: 53.1 %
# Accuracy of car: 74.9 %
# Accuracy of bird: 32.1 %
# Accuracy of cat: 23.2 %
# Accuracy of deer: 31.4 %
# Accuracy of dog: 37.9 %
# Accuracy of frog: 57.2 %
# Accuracy of horse: 67.9 %
# Accuracy of ship: 61.6 %
# Accuracy of truck: 47.8 %

# Finished Training - batch = 4
# Accuracy of the network: 50.13 %
# Accuracy of plane: 57.8 %
# Accuracy of car: 52.1 %
# Accuracy of bird: 25.0 %
# Accuracy of cat: 34.2 %
# Accuracy of deer: 51.2 %
# Accuracy of dog: 40.9 %
# Accuracy of frog: 64.6 %
# Accuracy of horse: 54.7 %
# Accuracy of ship: 58.6 %
# Accuracy of truck: 62.2 %

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
