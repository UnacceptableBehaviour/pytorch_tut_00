#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


#print(f"\n  \n{  }")
#print(f"\n  \n{  }")



# 0m  - intro to gradients
# 1m30 - requires_grad
print('- - - - - gradient of some function w/ respect to x')

x = torch.randn(3, requires_grad=True)  # tensor([-0.7058, -2.3361, -0.4301], requires_grad=True)
#x = torch.randn(3, requires_grad=False)  # causes RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

print(f"\n x = torch.randn(3, requires_grad=True)  \n{ x }")

# 1m30 - computational graph, forward pass, back propagation,
# FORWARD pass combine 2 inputs x & 2  through node(operation) to y
#
# BACKWARD pass
y = x + 2

print(f"\n y = x + 2 \n{ y }")          # tensor([ 3.1545, -0.0124,  1.0672], grad_fn=<AddBackward0>)
#                                                                                           ^
z = y*y*2
print(f"\n z = y*y*2 \n{ z }")          # tensor([10.2742,  8.1047,  4.3112], grad_fn=<MulBackward0>)
#                                                                                           ^

# 4m18 - on video grad_fn=<MeanBackward0>  but not here ? ? ? ?
z = z.mean()
print(f"\n z = z.mean() \n{ z }")       # 7.563368320465088
print('4m18 - on video grad_fn=<MeanBackward0>  but not here ? ? ? ?')

z.backward()
print(f"\n z.backward() \n{ x.grad }")  # tensor([2.0976, 3.5450, 1.6026])

# 5m30 -  vector Jacobean product. Jacobean matrix w/ derivatives, gradient vector = final gradients (chain rule)
# in above mean creates a scalra value no no argument is required for z.backward()
# for a tensor a vector Jacobiam Product required for z.backward()
z = y*y*2
vector_jacobian_product = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(vector_jacobian_product)
print(f"\n vector z.backward(vector_jacobian_product) \n{ x.grad }")  # tensor([2.0976, 3.5450, 1.6026])


# 8m20 -  preventing gradient tracking - 3 options
print('\n- - - - - preventing gradient tracking - 3 options')
# 1 - x.requires_grad_(False)
# 2 - x.detach()
# 3 - with torch.no_grad():

# 9m40 -  option 1
x = torch.randn(3, requires_grad=True)  # tensor([-0.7058, -2.3361, -0.4301], requires_grad=True)
print(f"\n1:  \n{ x }")
x.requires_grad_(False)   # note trailing _ modify in place
print(f"\n x.requires_grad_(False) \n{ x }")

# 10m15 -  option 2
x = torch.randn(3, requires_grad=True)  # tensor([-0.7058, -2.3361, -0.4301], requires_grad=True)
print(f"\n2:  \n{ x }")
y = x.detach()
print(f"\n y = x.detach() \n{ y }")

# 10m40 -  option 3
x = torch.randn(3, requires_grad=True)  # tensor([-0.7058, -2.3361, -0.4301], requires_grad=True)
print(f"\n3:  \n{ x }")
with torch.no_grad():
  y = x+2
  print(f"\n with torch.no_grad(): y=x+2 \n{ y }")

# 11m30 -  gradient accumulation / clearing
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
  model_output = (weights*3).sum()
  model_output.backward()
  print(f"\n for epoch in range(2): {epoch} \n{ weights.grad }")

weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
  model_output = (weights*3).sum()
  model_output.backward()
  print(f"\n for epoch in range(2): {epoch} w/ weights.grad.zero_()\n{ weights.grad }")
  weights.grad.zero_()      # zero out gradient so they don't accumulate

# 14m  -  optimiser - will do the step
# optimizer = torch.optim.SGD(weights, lr=0.01)   # SGD = stochastic gradient decent
# optimizer.step()
# optimizer.zero_grad()

# 15m  -  summary
# requires_grad must be set True for model building
# gradients ar calculated with .backward() call
# weights.grad.zero_() - zero out weight on each round -





#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
