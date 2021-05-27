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
#
#
#
#
#
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
op = torch.softmax(xx, dim=-2)   # dim=-2 ??
# dim=2 IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
print(f"\n torch.softmax(xx, dim=-2) \n{ op }")


#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
