#! /usr/bin/env python

# PyTorch Tutorial 07 - Linear Regression
# https://www.youtube.com/watch?v=YAJ5XBwlN4o&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=7

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

#### Steps in Torch ML pipeline
1) Design Model (input, output size, forward pass)
2) Construct the loss & optimiser
3) Training Loop
  - forward pass: compute prediction
  - backward pass: gradients
  - update weights

# 0m - review Steps in Torch ML pipeline
# 1m -
# 2m -

import torch
import torch.nn as nn       # PyTorch nn module has high-level APIs to build a neural network.
  # Torch. nn module uses Tensors and Automatic differentiation modules for training and building layers such as input,
  # hidden, and output layers - DOCS: https://pytorch.org/docs/stable/nn.html

import numpy as np      # NumPy is a library for the Python programming language, adding support for large,
  # multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate
  # on these arrays - DOCS: https://numpy.org/doc/stable/user/whatisnumpy.html

from sklearn import datasets  # to generate a regression dataset
                              # Scikit-learn is a library in Python that provides many unsupervised and supervised
  # learning algorithms. It contains a lot of efficient tools for machine learning and statistical modeling including
  # classification, regression, clustering and dimensionality reduction. Built upon some of the technology you might
  # already be familiar with, like NumPy, pandas, and Matplotlib!
  # DOCS: https://scikit-learn.org/stable/

import matplotlib.pyplot as plt # Matplotlib is a plotting library for the Python programming language. It provides an
  # object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter,
  # wxPython, Qt, or GTK - DOCS:
  # cheatsheets: https://github.com/matplotlib/cheatsheets#cheatsheets
  # How to plot & save graph hello world: https://github.com/UnacceptableBehaviour/latex_maths#python---matplotlib-numpy



#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
