#! /usr/bin/env python

# PyTorch Tutorial 11 - Softmax and Cross Entropy
# https://www.youtube.com/watch?v=7q7E91pHoW4&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=11

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - intro softmax maths
# 0m30 - softmax formula
# 1m20 - softmax diagram : scores, logits, probabilities (sum of probabilities = 1), prediction
# 1m40 - code start
# 2m - softmax: numpy
# 2m53 - softmax: torch
# 3m15 - Cross-Entropy: measures the performance of the output model
#      - better prediciotn = lower Cross-Entropy loss
# 4m05 - One-Hot encoding: Each class represented by a single binary 1 on classificaton array [0,1,0,0]
#      - line: TODO add number
# 4m30 - Y hat: predicted probablities (softmax)
# 5m17 - cross_entropy(Y actual, Y hat predicted)
# 6m50 - cross_entropy: torch caveats/gotchas slide
#      - nn.CrossEntropyLoss applies nn.LogSoftmax + nn NLLLoss (-ve log liklihood loss)
#      - Y has class labels not one-hot
#      - Y_pred (y hat) has raw scores (logits), no Softmax
# 7m50 - cross_entropy torch: code
#      - 3x rows
#14m10 - Neural Net w/ Softmax - slide - multi class
#15m05 - Neural Net w/ Softmax - code
#16m30 - Neural Net w/ Sigmoid - slide - binary
#17m10 - Neural Net w/ Sigmoid - code

#
# REFS
# softmax w/ 3d visuals
# https://www.youtube.com/watch?v=ytbYRIN0N4g
#
# Andrew Ng- walks through maths example
# https://www.youtube.com/watch?v=LLux1SW--oM
#
# Sigmoid function
# https://www.youtube.com/watch?v=TPqr8t919YM

# 2m
import torch
import torch.nn as nn
import torchvision
  # popular datasets, model architectures, and common image transformations for computer vision.
  # conda install torchvision -c pytorch    # already installed.

#from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])

print(f"\n np.x - {type(x)} \n{ x }")
print(f"\n np.exp(x) e={np.exp(1):.3f} \n{ np.exp(x) }")
print(f"\n np.sum(np.exp(x), axis=0) \n{ np.sum(np.exp(x), axis=0) }")
print(f"\n softmax(x) - numpy \n{ softmax(x) }")

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)   # dim=0 compute along the first axis

print(f"\n torch.softmax(x, dim=0) \n{ outputs }")


xx = torch.tensor([[2.0, 1.0, 0.1],
                  [2.9, 1.9, 0.3],
                  [8.0, 4.0, 0.4]])
op = torch.softmax(xx, dim=0)   # dim=0 compute along the first axis
print(f"\n torch.softmax(xx, dim=0) \n{ op }")

xx = torch.tensor([[2.0, 1.0, 0.1],
                  [2.9, 1.9, 0.3],
                  [8.0, 4.0, 0.4]])
op = torch.softmax(xx, dim=1)   # dim=1 compute ??
print(f"\n torch.softmax(xx, dim=1) \n{ op }")

xx = torch.tensor([[2.0, 1.0, 0.1],
                  [2.9, 1.9, 0.3],
                  [8.0, 4.0, 0.4]])
op = torch.softmax(xx, dim=-2)   # dim=-2 ?? TODO
# dim=2 IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
print(f"\n torch.softmax(xx, dim=-2) \n{ op }")


def cross_entropy(actual_Y, predicted_Yhat):
  loss = -np.sum(actual_Y * np.log(predicted_Yhat))
  return loss


# Y must be one hot encoded - for 8 classes an array of 8 elements is required
# 3 classes
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]

Y = np.array([1, 0, 0])                 # class 0
# EG predictions
Y_pred_good = np.array([0.7, 0.2, 0.1]) # of the 3 classes class 0 has the highest probability 0.7
                                        # example of a good predictor for  class 0
Y_pred_bad = np.array([0.1, 0.3, 0.6])  # of the 3 classes class 2 has the highest probability 0.6
                                        # example of a good predictor for class 0
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f"\n Loss l1 numpy \n{ l1 }")
print(f"\n Loss l2 numpy \n{ l2 }")

# 6m50 - cross_entropy - torch - caveats/gotchas
#      - nn.CrossEntropyLoss applies nn.LogSoftmax + nn NLLLoss (-ve log liklihood loss)
#      - Y has class labels not one-hot
#      - Y_pred (y hat) has raw scores (logits), no Softmax
#
# CrossEntropyLoss in PyTorch (applies Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = negative log likelihood loss
loss = nn.CrossEntropyLoss()         # passing callable instance class implements  __call__ method
# loss(input, target)

# target is of size nSamples = 1
# each element has class label: 0, 1, or 2
# Y (=target) contains class labels, not one-hot
Y = torch.tensor([0])

# input is of size nSamples x nClasses = 1 x 3
# y_pred (=input) must be raw, unnormalizes scores (logits) for each class, not softmax
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f"\n PyTorch Loss1: l1.item() \n{ l1.item() }")
print(f"\n PyTorch Loss2: l2.item() \n{ l2.item() }")

# TODO preeditions



#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
