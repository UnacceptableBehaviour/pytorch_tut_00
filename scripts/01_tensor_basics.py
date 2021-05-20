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




