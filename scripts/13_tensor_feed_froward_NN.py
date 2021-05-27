#! /usr/bin/env python

# PyTorch Tutorial 13 - Feed-Forward Neural Network
# https://www.youtube.com/watch?v=oPhxf2fXHkQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=13

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - intro to gradients
# 1m - theory - chain rule
# 2m - computational graph
# etc

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 0m - Overview:
# MNIST
# Dataloader, Transformation
# Multilayer NN Activation function
# Loss & Optimiser
# Training Loop
# Model Evaluation
# GPU support

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n device \n{ device }")

# Hyper-parameters
input_size = 784 # images 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

#print(f"\n  \n{  }")


# MNIST
# Dataloader, Transformation
# Multilayer NN Activation function
# Loss & Optimiser
# Training Loop
# Model Evaluation
# GPU support



#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
