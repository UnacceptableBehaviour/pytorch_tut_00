#! /usr/bin/env python

# PyTorch Tutorial 15 - Transfer Learning - using ResNet-18
# https://www.youtube.com/watch?v=K0lWSB2QoIQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=15
#
#  Transfer learning - tweaking last layers of a neural net to classify a new target (similar data type)
#  Used to specialize a large data set. Used on similar datasets: EG Images or sound
#
#  Examples - Ng
#  Fine tuning a 10K hours sound dataset neural net with 1hr of trigger-word data (Alexa)
#  or
#  Fine tuning a 1M image network with only 100 image of radiology image data
#
# see more here
# AndrewNg - Transfer Learning (C3W2L07)
# https://www.youtube.com/watch?v=yofjFQddwHE

# Resnet - residual neural network
# https://arxiv.org/pdf/1512.03385.pdf
# https://en.wikipedia.org/wiki/Residual_neural_network


#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - Transfer learning - tweaking last layers of a neural net to classify a new target (similar data type)
# 2m - Resnet - 1M images - 18layers
# 2m20 - Topics: Image folder, Scheduler, Transfer Learning
#
# 3m46 - Image folder structure
#
#   Scheduler
#
#  Transfer Learning
#
#


#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
