#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch
import numpy as np

print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


# 0m - Manual, Prediction, Gradient Computation, Loss Computation, Parameter updates
# 12m10 -
#
#
#
#
#
#
#
#
#
#
#

# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)
w = 0.0

# model predition
def forward(x):
  return w * x

# loss = MSE - Mean Squared Error
def loss(y, y_predicted):
  return ((y_predicted - y)**2).mean()

# gradient
#       mean   prediction   actual value   squared
# MSE = 1/N *  (w*x       -    y)           **2
# J - objective function
# dJ/dw = 1/N 2*x (w*x - y)
def gradient(x, y, y_predicted):
  return np.dot(2*x, y_predicted-y).mean()

print(f'prediction BEFORE training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 20 # 10

for epoch in range(n_iters):
  # prediction = forward pass
  y_pred = forward(X)

  # loss
  l = loss(Y, y_pred)

  # gradients
  dw = gradient(X, Y, y_pred)

  # update weights   -= negative gradient direction
  w -= learning_rate * dw

  if epoch % 1 ==  0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'prediction AFTER training: f(5) = {forward(5):.3f}')

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
