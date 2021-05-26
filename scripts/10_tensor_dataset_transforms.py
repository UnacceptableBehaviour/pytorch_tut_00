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
# 0m53 - Synopsis of transforms
# 1m40 - adapt WineDataset class
# 3m40 - custom transform class
# 6m50 - Mul transform class
# 8m50 - transform list

# List scarped with
# https://github.com/UnacceptableBehaviour/pytorch_tut_00/blob/main/scripts/fetch_transforms.py
#
# == COMPOSE MULTI TRANSFORM
# Compose(transforms)
#
# == ON IMAGES
# CenterCrop(size)
# ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
# FiveCrop(size)
# Grayscale(num_output_channels=1)
# Pad(padding, fill=0, padding_mode='constant')
# RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=<InterpolationMode.NEAREST: 'nearest'>, fill=0, fillcolor=None, resample=None)
# RandomApply(transforms, p=0.5)
# RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
# RandomGrayscale(p=0.1)
# RandomHorizontalFlip(p=0.5)
# RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=<InterpolationMode.BILINEAR: 'bilinear'>, fill=0)
# RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
# RandomRotation(degrees, interpolation=<InterpolationMode.NEAREST: 'nearest'>, expand=False, center=None, fill=0, resample=None)
# RandomSizedCrop(*args, **kwargs)
# RandomVerticalFlip(p=0.5)
# Resize(size, interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
# Scale(*args, **kwargs)
# TenCrop(size, vertical_flip=False)
# GaussianBlur(kernel_size, sigma=(0.1, 2.0))
# RandomChoice(transforms)
# RandomOrder(transforms)
#
# == ON TENSORS
# LinearTransformation(transformation_matrix, mean_vector)
# Normalize(mean, std, inplace=False)
# RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
#
# == CONVERSION
# ConvertImageDtype(dtype: torch.dtype) → None
# ToPILImage(mode=None)
# ToTensor
#
# == GENERIC
# Lambda(lambd)
#
# ==CUSTOM
#



import torch
import torchvision
  # popular datasets, model architectures, and common image transformations for computer vision.
  # conda install torchvision -c pytorch    # already installed.

# dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transform.ToTensor())

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
    self.x = xy[:, 1:]    # all ROWS :, from COL 1 onwards - 1:
    self.y = xy[:, [0]]   # all ROWS :, jsut COL 0         - [0]

    # use custom class (ToTensor below) to do this - from Ep 10
    # convert to torch
    # self.x = torch.from_numpy(xy[:, 1:])
    # self.y = torch.from_numpy(xy[:, [0]])

    self.n_samples = xy.shape[0]

    self.transform = transform


  def __getitem__(self, index):
    sample = self.x[index], self.y[index]

    if self.transform:
      sample = self.transform(sample)

    return sample

  def __len__(self):
    return self.n_samples


# custom class to convert from numpy to tensor
class ToTensor:
  print('ToTensor - class var scope')
  # The __call__ method enables Python programmers to write classes where the instances behave
  # like functions and can be called like a function.
  # EGs in
  # https://github.com/UnacceptableBehaviour/python_koans/blob/master/python3/scratch_pad_1b_instance__call__.py
  def __call__(self, sample):
    inputs, targets = sample
    return torch.from_numpy(inputs), torch.from_numpy(targets)


# t = ToTensor()  # t =  ToTensor callable object
# t()             # TypeError: __call__() missing 1 required positional argument: 'sample'


dataset = WineDataset(transform=None)
fisrt_data = dataset[0]
features, labels = fisrt_data
print(f"\n type(features) - transform=None \n{ type(features) }") # <class 'numpy.ndarray'>
print(f"\n type(labels)   - transform=None \n{ type(labels) }")   # <class 'numpy.ndarray'>

dataset = WineDataset(transform=ToTensor())
fisrt_data = dataset[0]
features, labels = fisrt_data
print(f"\n type(features) - transform=ToTensor()\n{ type(features) }")    # <class 'torch.Tensor'>
print(f"\n type(labels)   - transform=ToTensor()\n{ type(labels) }")      # <class 'torch.Tensor'>








#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
