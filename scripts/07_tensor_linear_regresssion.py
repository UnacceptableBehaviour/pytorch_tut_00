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
# 1) Design Model (input, output size, forward pass)
# 2) Construct the loss & optimiser
# 3) Training Loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

# 0m - review Steps in Torch ML pipeline
# 1m - library imports
# 2m - coding starts

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


# 0) prepare data - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
#          returned from  ^
# change data type from double to float32 - avoid erros later
X = torch.from_numpy(x_numpy.astype(np.float32))                # create torch tensor from numpy array
Y = torch.from_numpy(y_numpy.astype(np.float32))
print(f"\n Y = torch.from_numpy(y_numpy.astype(np.float32)) \n{ Y }")
#  Y = torch.from_numpy(y_numpy.astype(np.float32))             # tensor w a single row - see square brackets
# tensor([-5.5539e+01, -1.0662e+01,  2.2757e+01,  1.0110e+02,  1.4434e+02,
#          3.3289e+01,  3.3015e+01, -2.5887e+01, -9.9639e+01,  2.3803e+01,
#         -4.5589e+01, -8.3388e+00, -9.5315e+01,  3.6407e+01, -8.7293e+01,
#          6.7669e+01, -1.3687e+01, -5.5441e+01, -6.5340e+01, -5.4450e+01,
#         -2.8835e+01,  1.7884e+02,  6.5084e+01,  2.6668e+01, -1.8546e+01,
#         -4.1499e+01,  8.5583e-01,  4.4562e+01,  1.1598e+02, -6.4620e+01,
#         -2.5931e+01, -6.0882e+01,  1.8720e+01,  7.5070e+01,  1.1720e+02,
#         -2.2698e+01, -5.6363e+01,  1.8084e+02, -1.9257e+02,  6.8503e+01,
#          1.6552e+02,  1.0500e+02, -7.0434e+01, -5.8769e+01, -4.1576e+01,
#          7.3247e+01,  4.0966e+01,  8.0462e+01, -2.8794e+01,  3.4234e+01,
#         -4.1715e+01,  1.4355e+01,  7.9336e+01,  2.7129e+01, -3.9487e+01,
#          6.6805e+01,  9.5531e+01,  3.5610e+00,  1.0857e-01,  5.6495e+01,
#          5.1575e+01, -2.0974e+00, -2.6656e+01,  3.9742e+01,  3.6101e+01,
#         -7.5602e+01,  1.9713e+01, -7.1601e+01, -1.9904e+01, -7.6708e+01,
#         -1.1834e+02, -2.9825e+01,  1.5108e+02,  5.2923e+01, -5.9552e+01,
#          3.0721e+01, -2.9355e+01, -4.4786e+01,  1.0006e+02,  1.5058e+02,
#          1.2200e+02, -1.8186e+02,  3.4739e+00, -2.2980e+01,  4.5184e+01,
#          9.8606e+01, -9.2779e+00, -5.2478e+01,  3.8593e+01, -1.9997e+02,
#         -9.5201e+00, -3.4724e+00, -3.5312e+01,  7.5406e+01,  1.7570e+01,
#         -2.3960e+01,  1.3209e+02,  2.0608e+01,  5.1111e+01, -2.6306e+01])

print(f"\n Y.shape[0] \n{ Y.shape[0] }")  # 100
y = Y.view(Y.shape[0], 1)                       # reshape to a column tensor Y.view(ROW, COL) Y.view(100, 1)
print(f"\n y = Y.view(y.shape[0], 1) \n{ y }")

# tensor([[-5.5539e+01],
#         [-1.0662e+01],
#         [ 2.2757e+01],
#         [ 1.0110e+02],
#         .
#         100 in total
#         .
#         [ 1.3209e+02],
#         [ 2.0608e+01],
#         [ 5.1111e+01],
#         [-2.6306e+01]])

print(f"\n y.shape \n{ y.shape }")      # new little y shape = torch.Size([100, 1])  ROWS, COLS
print(f"\n X.shape \n{ X.shape }")
n_samples, n_features = X.shape




#print(f"\n  \n{  }")

# 1) model - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# in LINEAR REGRESSION case this is ONE layer
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)    # built in  Linear model

# 2) loss optimizer - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
learning_rate = 0.01

criterion = nn.MSELoss()        # for LINEAR REGRESSION - BUILT IN Loss function Mean Squared Error Loss
  # nn.MSELoss() creates a criterion - https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # SGD - Stocastic Gradient Descent
  # https://pytorch.org/docs/stable/optim.html?highlight=torch%20optim%20sgd#torch.optim.SGD
  # w/ optional Nesterov momentum :o

# 3) training loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
num_epochs = 100

for epoch in range(num_epochs):
  #   - forward pass: compute prediction
  y_predicted = model(X)              # call model passing in data X
  loss = criterion(y_predicted, y)    # actual labels & predicted   - output = criterion(input, target)

  #   - backward pass: gradients
  loss.backward()

  #   - update weights
  optimizer.step()
  optimizer.zero_grad()

  if (epoch+1) % 10 == 0:
    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()     # prevent gradient tracking?
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show
print('plt.show')
print(f"\n x_numpy \n{ x_numpy }")
print(f"\n y_numpy \n{ y_numpy }")
print(f"\n predicted \n{ predicted }")



#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
