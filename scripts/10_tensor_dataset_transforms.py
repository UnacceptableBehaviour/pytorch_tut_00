#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - intro to transforms Link: https://pytorch.org/vision/stable/transforms.html
# source - https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html
# 1m40 - adapt WineDataset class
# 3m40 - custom transform class
# 6m50 - Mul transform class
# 8m50 - transform list


import torch
import torchvision
  # popular datasets, model architectures, and common image transformations for computer vision.
  # conda install torchvision -c pytorch    # already installed.

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math



class WineDataset(Dataset):
  # each ROW is a sample
  def __init__(self, transform=None):
    # x - features
    # y - class - wine in this case

    # * * * N O T E * * *
    # run ./scripts/10_tensor_dataset_transforms.py from /pytorch/pytorch_tut_00
    xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1) # skiprows=1 - skip header row

    # split dataset into x & y
    # self.x = xy[:, 1:]    # all ROWS :, from COL 1 onwards - 1:
    # self.x = xy[:, [0]]   # all ROWS :, jsut COL 0         - [0]
    # convert to torch

    self.x = torch.from_numpy(xy[:, 1:])
    self.y = torch.from_numpy(xy[:, [0]])

    self.n_samples = xy.shape[0]

    self.transform =  transform



  def __getitem__(self, index):
    sample = self.x[index], self.y[index]

    if self.transform:
      sample = self.transform(sample)

    return sample

  def __len__(self):
    return self.n_samples

dataset = WineDataset()

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
