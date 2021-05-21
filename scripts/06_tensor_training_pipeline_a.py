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
# 2m40 - Step 3 - Adapt code - import torch.nn
# 5m04 - introduce a pytorch model
# 11m40 - custom LinearRegression model

# 0m - Steps in Torch ML pipeline
# 1) Design Model (input, output size, forward pass)
# 2) Construct the Loos & optimiser
# 3) Training Loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights



# f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)  # number of row = number of samples
                                                          # features
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
print(f'X_test {X_test}')

print(f'X.shape {X.shape}')
n_samples, n_features = X.shape
print(f'n_samples {n_samples}, n_features {n_features}')

input_size = n_features
output_size = n_features
# normally we would have to design this. But since this example is trivial can use built ins
# model = nn.Linear(input_size, output_size)     # <
                                                 #
# even after 1000 epoch not quite there!         #
#                                                #
# X_test tensor([5.])                            #
# X.shape torch.Size([4, 1])                     #
# n_samples 4, n_features 1                      #
# prediction BEFORE training: f(5) = 4.837       #
# epoch 1: w = 0.957, loss = 6.61540985          #
# epoch 11: w = 1.548, loss = 0.33475524         #
# epoch 21: w = 1.652, loss = 0.16273710         #
# .                                              #
# .                                              #
# epoch 971: w = 1.981, loss = 0.00053171        #
# epoch 981: w = 1.981, loss = 0.00050076        #
# epoch 991: w = 1.982, loss = 0.00047161        #
# prediction AFTER training: f(5) = 9.964        #
#                                                #
# tried 10000 - model crystalised after 2711     #
# epoch 2711: w = 2.000, loss = 0.00000001       #
# epoch 2721: w = 2.000, loss = 0.00000000       #


# custom model
class LinearRegression(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(LinearRegression, self).__init__()
    # define layers
    self.lin = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.lin(x)

model = LinearRegression(input_size, output_size)

# X_test tensor([5.])
# X.shape torch.Size([4, 1])
# n_samples 4, n_features 1
# prediction BEFORE training: f(5) = 3.426
# epoch 1: w = 0.727, loss = 11.25522518
# epoch 11: w = 1.499, loss = 0.46884847
# .
# . 1
# epoch 2911: w = 2.000, loss = 0.00000001
# epoch 2921: w = 2.000, loss = 0.00000000
# . 2
# epoch 3011: w = 2.000, loss = 0.00000001
# epoch 3021: w = 2.000, loss = 0.00000000
# . 3
# epoch 2101: w = 2.000, loss = 0.00000001
# epoch 2111: w = 2.000, loss = 0.00000000
# . 4
# epoch 2131: w = 2.000, loss = 0.00000001
# epoch 2141: w = 2.000, loss = 0.00000000

print(f'prediction BEFORE training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_iters = 10000 # 10

loss = nn.MSELoss()   # Mean Squared Error Loss    -  REMOVE brackets?
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent


for epoch in range(n_iters):
  # prediction = forward pass
  y_pred = model(X)

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
    [w, b] = model.parameters() # unpack weights & bias optional
                                # w - list of lists
    print(f'epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}')


print(f'prediction AFTER training: f(5) = {model(X_test).item():.3f}')

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
