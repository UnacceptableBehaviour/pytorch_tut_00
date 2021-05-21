#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch

print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


# 0m - Manual, Prediction, Gradient Computation, Loss Computation, Parameter updates
# 12m10 - switch over from numpy to torch

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model predition
def forward(x):
  return w * x

# loss = MSE - Mean Squared Error
def loss(y, y_predicted):
  return ((y_predicted - y)**2).mean()


print(f'prediction BEFORE training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 80 # 10

for epoch in range(n_iters):
  # prediction = forward pass
  y_pred = forward(X)

  # loss
  l = loss(Y, y_pred)

  # gradients = backward pass
  l.backward()  # dl/dw - gradient of loss wrt weights
                # accumulate gradients in w.grad attirbute

  # update weights
  with torch.no_grad():
    w -= learning_rate * w.grad

  # zero out gradients
  w.grad.zero_()

  if epoch % 10 ==  0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'prediction AFTER training: f(5) = {forward(5):.3f}')

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
