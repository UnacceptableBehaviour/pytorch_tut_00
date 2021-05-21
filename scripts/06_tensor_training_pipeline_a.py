#! /usr/bin/env python

# PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer
# https://www.youtube.com/watch?v=VVDHU_TWwUg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=6 pt 2

#from __future__ import print_function
import torch
import torch.nn as nn

print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


# 0m - Steps in Torch ML pipeline
# 2m40 - Adapt code
#
#
#
#
#
#
#
#
#

# 0m - Steps in Torch ML pipeline
# 1) Design Model (input, output size, forward pass)
# 2) Contsruct the Loos & optimiser
# 3) Training Loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights



# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # weights

# model predition
def forward(x):
  return w * x


print(f'prediction BEFORE training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iters = 80 # 10

loss = nn.MSELoss()   # Mean Squared Error Loss    -  REMOVE brackets?
optimizer = torch.optim.SGD([w], lr=learning_rate)


for epoch in range(n_iters):
  # prediction = forward pass
  y_pred = forward(X)

  # loss
  l = loss(Y, y_pred)

  # gradients = backward pass
  l.backward()  # dl/dw - gradient of loss wrt weights
                # accumulate gradients in w.grad attirbute

  # update weights
  optimizer.step()

  # zero out gradients
  optimizer.zero_grad()

  if epoch % 10 ==  0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'prediction AFTER training: f(5) = {forward(5):.3f}')

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
