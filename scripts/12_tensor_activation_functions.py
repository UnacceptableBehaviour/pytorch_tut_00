#! /usr/bin/env python

# PyTorch Tutorial 12 - Activation Functions
# https://www.youtube.com/watch?v=3t9lZM7SS7k&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=12

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - Intro to activation functions - rationale
# 2m - Popular activation functions
#    - Step function, Sigmoid, TanH, ReLU, Leaky ReLU, Softmax
# 2m25 - Sigmoid 0 to 1: Last layer of binary classificatioon problem
# 2m50 - TanH -1 to +1: Scaled & shifted sigmoid function - used in hidden layers
# 3m20 - ReLU: Rectified Linear Unit: 0 for -ve inputs, linear for +ve inputs
#      - better performance than sigmoid
# 4m20 - Leaky ReLU: Used in solving Vanishing gradient problem
# 5m40 - Softmax:Typically good choice in last layer of a multi classification problem
# 6m30 - Walk the 1st Neural Network code
# 7m40 - Walk the 2nd Neural Network code
# note - the NN code isn't executed - next episode
# 8m30 - API: torch.nn, torch.nn.functional


#print(f"\n  \n{  }")
#print(f"\n  \n{  }")

# output = w*x + b
# output = activation_function(output)
import torch
import torch.nn as nn
import torch.nn.functional as F

def dmpAcc(op_tensor):
    t = 0
    for i in op_tensor:
        t += i
        print(f"i = {i:.3f} - total: {t:.3f} ")


x = torch.tensor([-1.0, 1.0, 2.0, 3.0])
print(f"\n torch.tensor([-1.0, 1.0, 2.0, 3.0]) \n{ x }")

#### Softmax
output = torch.softmax(x, dim=0)
print(f"\n - - - - torch.softmax(x, dim=0) \n{ output }")
dmpAcc(output)

sm = nn.Softmax(dim=0)
output = sm(x)
print(f"\n sm = nn.Softmax(dim=0) sm(x) \n{ output }")

#### Sigmoid,
output = torch.sigmoid(x)
print(f"\n - - - - torch.sigmoid(x) \n{ output }")
dmpAcc(output)

s = nn.Sigmoid()
output = s(x)
print(f"\n s = nn.Sigmoid() s(x) \n{ output }")


#### TanH,
output = torch.tanh(x)
print(f"\n - - - - torch.tanh(x) \n{ output }")
dmpAcc(output)

t = nn.Tanh()
output = t(x)
print(f"\n  \n{ output }")

#### ReLU,
output = torch.relu(x)
print(f"\n - - - - torch.relu(x)  \n{ output }")
dmpAcc(output)

relu = nn.ReLU()
output = relu(x)
print(f"\n  \n{ output }")

#### Leaky ReLU,
output = F.leaky_relu(x)
print(f"\n - - - - F.leaky_relu(x) F = torch.nn.functional  \n{ output }")
dmpAcc(output)

lrelu = nn.LeakyReLU()
output = lrelu(x)
print(f"\n  \n{ output }")

#nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
#torch.relu on the other side is just the functional API call to the relu function,
#so that you can add it e.g. in your forward method yourself.

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    # initialise models named object vasr with standard functions
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    # use object name in forward pass
    # output from each step/layer is passed to the next step/layer
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation functions directly in forward pass)
# basicall a bit juggled about
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


print('\n')
