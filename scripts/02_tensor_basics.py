#! /usr/bin/env python

# PyTorch Tutorial 02 - Tensor Basics
# https://www.youtube.com/watch?v=exaWOE8jvy8&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=2


#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

print("\n1")
x = torch.empty(1)
print(x)

print("\ntorch.empty(3)")
x = torch.empty(3)
print(x)

print("\n2d 2,3")
x = torch.empty(2,3)
print(x)

print("\n3d torch.empty(2,3,4)")
x = torch.empty(2,3,4)
print(x)

print("\n4d 2,3,4,3 - torch.empty(2,3,4,3)")
x = torch.empty(2,3,4,3)
print(x)


print("\n3d 2,3,4 - torch.zeros(2,3,4) - INITIALIZED")
x = torch.zeros(2,3,4)
print(x)

a = torch.ones(5)
print("\ntorch.ones(5) - INITIALIZED ones")
print(a)

print("\n3d 2,3,4 - torch.rand(2,3,4) - initialized RANDOM")
x = torch.rand(2,3,4)
print(x)

print('\ntypes')

x = torch.empty(2,3)
print(f"python {x.type}")
print(f"tensor {x.dtype}")


print(f"python {x.type}")
print(f"tensor {x.dtype}")
x = torch.empty(2,3, dtype=torch.float16)
y = torch.empty(2,3, dtype=torch.double)
z = torch.empty(2,3, dtype=torch.int)
print(f"tensor x float16 {x.dtype}")
print(f"tensor y double  {y.dtype}")
print(f"tensor z int     {z.dtype}")

print(f"\nsize {x.size()} dimensions")

print(f"\naddition\n {x + y} ")
print(torch.add(x,y))


print('\n* * * in place addition y.add_(x) the underscore = result stored in y         < **')
print('y.add_(x)  < **\n')
print(y.add_(x))
print('\n\n')

# 3m25

a = ['aple','banan','orang','betrut']
#mishapes = torch.tensor(a)    # ValueError: too many dimensions 'str'
# STRINGS - NO!

print('tensor from list')
b = [2.5, 3.0, 26.3]
mishapes = torch.tensor(b)    # OK

print(mishapes)

x = torch.rand(2,2)
y = torch.rand(2,2)
print('\n- - - - - Rithmatic init + - * /')
print('x\n',x)
print('y\n',y)
print('x+y\n',x+y)
print(f"\n=- addition: x+y = torch.add(x,y)\n {x+y}\n{torch.add(x,y)}")
print(f"\n=- subtraction: x-y = torch.sub(x,y)\n {x+y}\n{torch.sub(x,y)}")
print(f"\n=- mult: x*y = torch.mul(x,y)\n {x+y}\n{torch.mul(x,y)}")
print(f"\n=- div: x/y = torch.div(x,y)\n {x/y}\n{torch.div(x,y)}")
print(f"\n=- ad const: x + 1   x.add_(1) add _ in place \n {x.add_(1)}")
#print(f"\n=- ad const: x + 1   x += 1          in place \n {x += 1}")

# 8m11
print('\n- - - - - Manipulations & Type conversion')
x = torch.rand(5,3)
print(f"x = torch.rand(5,3)\n{x}")
print(f"\nslice: x[:,0] = : all rows, column 0\n {x[:,0]}")
print(f"\nslice: x[1,:] = : all column, rows  1\n {x[1,:]}")
print(f"\nexact element: x[4,2] = : row 4, column 2  1\n {x[4,2]}")
print('* * * NOTE x[ROW,COL] not cartesian INVERSE from expected')

# 10m25 resizing
x = torch.rand(2,6)
print('\n\n10m25 resizing')
print(f"x = torch.rand(2,6) 2 ROWS, 6 COLS\n{x}")

y = x.view(12)
print(f"\nx.view(12) - 1 dimension - same number  of elements\n{y}")

y = x.view(-1,4)
print(f"\nx.view(-1,4) - specifying COLS only\n{y}")

y = x.view(4,-1)
print(f"\nx.view(4,-1) - specifying ROWS only\n{y}")
print(f"\ny.size() - get new size\n{y.size()}")

# 10m44 converting from numpy to tensor
print('\n\n10m44 converting from numpy to tensor')

import numpy as np
a = torch.ones(5)
print("\ntorch.ones(5) - INITIALIZED ones - CONVERTING from TORCH to NUMPY")
print(a)
b = a.numpy()   # <class 'numpy.ndarray'>
print(f"\nb = a.numpy() type(b)\n{type(b)}")



# 11m50 tensor by refence issues / numpy cpu vs gpu
# comment about gpu & cpu
# creating a numy array from a tenso & vice versa
print('\n\n* * * WARNING - 11m50 - if the TENSOR on CPU & not the GPU \n then BOTH objects will SHARE the SAME MEMORY LOCATION')
print("\nb = a.numpy()")
print(f"\n a \n{ a }")      # tensor([1., 1., 1., 1., 1.])
print(f"\n b \n{ b }")      # [1. 1. 1. 1. 1.]
b[2] = 5                    # tensor([1., 1., 5., 1., 1.])
print(f"\n b[2] = 5 \n{ b }")
print(f"\n a \n{ a }")      # tensor([1., 1., 5., 1., 1.])
print(f"\n b \n{ b }")      # [1. 1. 5. 1. 1.]



print("\n - - - - - CONVERTING from NUMPY to TORCH")
a = np.ones(5)
print(f"\n a = np.ones(5) \n{ a }")               # [1. 1. 1. 1. 1.]
b = torch.from_numpy(a)
print(f"\n b = torch.from_numpy(a) \n{ b }")      # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

print('\n - - WARNING - same issue')
a += 1
print('a += 1')
print(f"\n a \n{ a }") # [2. 2. 2. 2. 2.]
print(f"\n b \n{ b }") # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

# 14m40 creating tensor on the GPU
if torch.cuda.is_available():
  device = troch.device("cuda")
  x = torch.ones(5, device=device)  # create tensor on GPU
  y = torch.ones(5)                 # create tensor on CPU
  y = y.to(device)                  # move tensor to GPU
  z = x + y                         # operate on GPU
  # z.numpy()  ERROR numpy CPU only - convert to CPU 1st
  z = z.to("cpu")                   # 17m moving to GPU and back to CPU
  z.numpy()

# 17m moving to GPU and back to CPU

# 17m20 requires_grad
x = torch.ones(5, requires_grad = True)   # high light that tensor will need to calculate gradients later - see next vid

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")



