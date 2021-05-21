#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch





# 13m15

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
  model_output = (weights*3).sum()

  model_output.backward()

  print(weights.grad)
