#! /usr/bin/env python

#from __future__ import print_function
import torch

print("\n1")
x = torch.empty(1)
print(x)

print("\n3")
x = torch.empty(3)
print(x)

print("\n2d 2,3")
x = torch.empty(2,3)
print(x)

print("\n3d 2,3,4")
x = torch.empty(2,3,4)
print(x)

print("\n4d 2,3,4,3")
x = torch.empty(2,3,4,3)
print(x)


print("\n3d 2,3,4")
x = torch.zeros(2,3,4)
print(x)

print("\n3d 2,3,4")
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

print('\nin place addition y.add_(x) the underscore = result stored in y         < **')
print(y.add_(x))

# 3m25

a = ['aple','banan','orang','betrut']
#mishapes = torch.tensor(a)    # ValueError: too many dimensions 'str'
# STRINGS - NO!

b = [2.5, 3.0, 26.3]
mishapes = torch.tensor(b)    # OK

print(mishapes)

x = torch.rand(2,2)
y = torch.rand(2,2)
print('\n- - - - - Rithmatic init + - * /')
print('x\n',x)
print('y\n',y)
print('x+y\n',x+y)
print(f"\naddition: x+y = torch.add(x,y)\n {x+y}\n{torch.add(x,y)}")
print(f"\nsubtraction: x-y = torch.sub(x,y)\n {x+y}\n{torch.sub(x,y)}")
print(f"\nmult: x*y = torch.mul(x,y)\n {x+y}\n{torch.mul(x,y)}")
print(f"\ndiv: x/y = torch.div(x,y)\n {x/y}\n{torch.div(x,y)}")

print('\n- - - - - Manipulations')
x = torch.rand(5,3)
print(f"rand[5,3]\n{x}")
print(f"\nslice: x[:,0] = : all rows, column 0\n {x[:,0]}")


# 10m44 resizing

# 11m tensor by refence issues / numpy cpu vs gpu
# comment about gpu & cpu
# creating a numy array from a tenso & vice versa

# 17m moving to GPU and back to CPU





