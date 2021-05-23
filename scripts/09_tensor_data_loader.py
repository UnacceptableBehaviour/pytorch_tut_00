#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - intro to dataloader classes
# 1m30 - terms: epoch, batch size, num of iteration,
# 2m - libs: torch, torchvision, torch.utils.data, numpy, math
# 2m30 - Custom datase
# 7m13 - completed class WineDataset(Dataset)
# 7m30 - inspect dataset
# 10m18 - feature vectors & class label ispect
# 10m40 - training loop - iterating dataset
# 14m50 - example data sets: MNIST et al

# ```
# epoch = 1 forward & backward pass of all training samples
# batch_size = number of training samples in forward & backward pass
# number of iterations = number of passes, each pass using [batch_size] number of samples
# EG 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch
#
# in the data each ROW is a sample
# ```

import torch
import torchvision
  # popular datasets, model architectures, and common image transformations for computer vision.
  # conda install torchvision -c pytorch    # already installed.

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

from sklearn.datasets import load_wine
data = load_wine()

print(data)
# {'data': array([[1.423e+01, 1.710e+00, 2.430e+00, ..., 1.040e+00, 3.920e+00,
#         1.065e+03],
#        [1.320e+01, 1.780e+00, 2.140e+00, ..., 1.050e+00, 3.400e+00,
#         1.050e+03],
#        [1.316e+01, 2.360e+00, 2.670e+00, ..., 1.030e+00, 3.170e+00,
#         1.185e+03],
#        ...,
#        [1.327e+01, 4.280e+00, 2.260e+00, ..., 5.900e-01, 1.560e+00,
#         8.350e+02],
#        [1.317e+01, 2.590e+00, 2.370e+00, ..., 6.000e-01, 1.620e+00,
#         8.400e+02],
#        [1.413e+01, 4.100e+00, 2.740e+00, ..., 6.100e-01, 1.600e+00,
#         5.600e+02]]),
#  'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2]),
#  'frame': None,
#  'target_names': array(['class_0', 'class_1', 'class_2'], dtype='<U7'),
#  'DESCR': '.. _wine_dataset:\n\nWine recogn . . . . ~3600 cols - see print(data.DESCR) below
# 'feature_names': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
#                   'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
# }

print('\ndata.DESCR')
print(data.DESCR)
print('\ndata.data')
print(data.data)
print('\ndata.target_names')
print(data.target_names)
print('\ndata.feature_names')
print(data.feature_names)

# print(data.DESCR)
#
# .. _wine_dataset:
#
# Wine recognition dataset
# ------------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 178 (50 in each of three classes)
#     :Number of Attributes: 13 numeric, predictive attributes and the class
#     :Attribute Information:
#  		- Alcohol
#  		- Malic acid
#  		- Ash
# 		- Alcalinity of ash
#  		- Magnesium
# 		- Total phenols
#  		- Flavanoids
#  		- Nonflavanoid phenols
#  		- Proanthocyanins
# 		- Color intensity
#  		- Hue
#  		- OD280/OD315 of diluted wines
#  		- Proline
#
#     - class:
#             - class_0
#             - class_1
#             - class_2
#
#     :Summary Statistics:
#
#     ============================= ==== ===== ======= =====
#                                    Min   Max   Mean     SD
#     ============================= ==== ===== ======= =====
#     Alcohol:                      11.0  14.8    13.0   0.8
#     Malic Acid:                   0.74  5.80    2.34  1.12
#     Ash:                          1.36  3.23    2.36  0.27
#     Alcalinity of Ash:            10.6  30.0    19.5   3.3
#     Magnesium:                    70.0 162.0    99.7  14.3
#     Total Phenols:                0.98  3.88    2.29  0.63
#     Flavanoids:                   0.34  5.08    2.03  1.00
#     Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
#     Proanthocyanins:              0.41  3.58    1.59  0.57
#     Colour Intensity:              1.3  13.0     5.1   2.3
#     Hue:                          0.48  1.71    0.96  0.23
#     OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
#     Proline:                       278  1680     746   315
#     ============================= ==== ===== ======= =====
#
#     :Missing Attribute Values: None
#     :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
#     :Creator: R.A. Fisher
#     :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
#     :Date: July, 1988
#
# This is a copy of UCI ML Wine recognition datasets.
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
#
# The data is the results of a chemical analysis of wines grown in the same
# region in Italy by three different cultivators. There are thirteen different
# measurements taken for different constituents found in the three types of
# wine.
#
# Original Owners:
#
# Forina, M. et al, PARVUS -
# An Extendible Package for Data Exploration, Classification and Correlation.
# Institute of Pharmaceutical and Food Analysis and Technologies,
# Via Brigata Salerno, 16147 Genoa, Italy.
#
# Citation:
#
# Lichman, M. (2013). UCI Machine Learning Repository
# [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
# School of Information and Computer Science.
#
# .. topic:: References
#
#   (1) S. Aeberhard, D. Coomans and O. de Vel,
#   Comparison of Classifiers in High Dimensional Settings,
#   Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of
#   Mathematics and Statistics, James Cook University of North Queensland.
#   (Also submitted to Technometrics).
#
#   The data was used with many others for comparing various
#   classifiers. The classes are separable, though only RDA
#   has achieved 100% correct classification.
#   (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data))
#   (All results using the leave-one-out technique)
#
#   (2) S. Aeberhard, D. Coomans and O. de Vel,
#   "THE CLASSIFICATION PERFORMANCE OF RDA"
#   Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of
#   Mathematics and Statistics, James Cook University of North Queensland.
#   (Also submitted to Journal of Chemometrics).


features, target = load_wine(return_X_y=True)
print('\n features')
print(features)
#  features
# [[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
#  [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
#  [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
#  ...
#  [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]
#  [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]
#  [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]


print('\n target')
print(target)
#  target
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# print('- - - iterating data')
# for i in data.data.item():
#   print(i)

print('\n\n\n** L O A D I N G   F R O M   C S V   F I L E **\n\n\n')

class WineDataset(Dataset):
  # each ROW is a sample
  def __init__(self):
    # x - features
    # y - class - wine in this case

    # * * * N O T E * * *
    # run ./scripts/09_tensor_data_loader.py from /pytorch/pytorch_tut_00
    xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1) # skiprows=1 - skip header row

    # split dataset into x & y
    # self.x = xy[:, 1:]    # all ROWS :, from COL 1 onwards - 1:
    # self.x = xy[:, [0]]   # all ROWS :, jsut COL 0         - [0]
    # convert to torch

    self.x = torch.from_numpy(xy[:, 1:])
    self.y = torch.from_numpy(xy[:, [0]])

    self.n_samples = xy.shape[0]


  def __getitem__(self, index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples

dataset = WineDataset()

print(dataset)

first_data = dataset[0]
features, labels = first_data
print(f"\n features \n{ features }")
print(f"\n labels \n{ labels }")

b_size = 6
dataloader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=True, num_workers=2) # num_workers - use mor than one process
#                                            ^            ^
#                     how many to get per fetch          shuffle data on fetch

dataIter = iter(dataloader)
data = dataIter.next()
features, labels = data
print(f"\n iter features batch_size:{b_size} \n{ features }")
print(f"\n iter labels batch_size:{b_size} \n{ labels }")

b_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=True, num_workers=2)
dataIter = iter(dataloader)

# for batch in dataIter:
#   features, labels = batch
#   print('\nbatch - - - \\')
#   print(f"\n iter features batch_size:{b_size} \n{ features }")
#   print(f"\n iter labels batch_size:{b_size} \n{ labels }")


# training loop

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

print(f"\n total_samples \n{ total_samples }")
print(f"\n n_iterations \n{ n_iterations }")

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # 178 rows > 178 samples
        # i will count them out
        # input - x - features
        # labels - y - wine 1,2,3
        # iterations 178 sample / batch size 4 . . ceil(44.5) = 45

        # Run your training process
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_iterations} \t inputs {inputs.shape} \t labels {labels.shape}')

#  total_samples
# 178
#
#  n_iterations
# 45
# epoch: 1/2, step 5/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 10/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 15/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 20/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 25/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 30/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 35/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 40/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 1/2, step 45/45 	 inputs torch.Size([2, 13]) 	 labels torch.Size([2, 1])
# epoch: 2/2, step 5/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 10/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 15/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 20/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 25/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 30/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 35/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 40/45 	 inputs torch.Size([4, 13]) 	 labels torch.Size([4, 1])
# epoch: 2/2, step 45/45 	 inputs torch.Size([2, 13]) 	 labels torch.Size([2, 1])




# some FAMOUS datasets are available in torchvision.datasets
# e.g. MNIST, fashion-mnist, cifar10, coco

train_dataset = torchvision.datasets.MNIST(root='./scratch',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=3,
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
inputs, targets = data
print(f"\n inputs.shape \n{ inputs.shape }")
print(f"\n targets.shape \n{ targets.shape }")
#  inputs.shape
# torch.Size([3, 1, 28, 28])
#
#  targets.shape
# torch.Size([3])

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
