#! /usr/bin/env python

# PyTorch Tutorial 04 - Backpropagation - Theory With Example
# https://www.youtube.com/watch?v=3Kb0QS6z7WA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=4

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


# 0m - intro to gradients
# 1m - theory - chain rule
# 2m - computational graph
# 4m12 - backward pass - concept overview
# 4m30 - linear regression
# 4m50 -  walk through the maths
# 10m30 - pytorch implementation

# y hat - predicted loss

x = torch.tensor(1.0)
print(f"\n torch.tensor(1.0) \n{ x }")
y = torch.tensor(2.0)
print(f"\n torch.tensor(2.0) \n{ y }")

w = torch.tensor(1.0, requires_grad=True)
print(f"\n w = torch.tensor(1.0, requires_grad=True) \n{ w }")


y_hat = w * x
print(f"\n y_hat = w * x \n{ y_hat }")

loss = (y_hat - y)**2
print(f"\n loss = (y_hat - y)**2 \n{ loss }")

print('- - - - - backward pass')
loss.backward()
print(f"\n loss.backward() (w.grad)\n{ w.grad }")

# update weights
# next forward pass & back prpagate . . . loop

#print(f"\n  \n{  }")
print('\n')
