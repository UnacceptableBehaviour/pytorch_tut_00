#! /usr/bin/env python

# PyTorch Tutorial 13 - Feed-Forward Neural Network
# https://www.youtube.com/watch?v=oPhxf2fXHkQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=13

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 0m - Overview:
# MNIST
# Dataloader, Transformation
# Multilayer NN Activation function
# Loss & Optimiser
# Training Loop
# Model Evaluation
# GPU support
# 1m30 - Hyperparameters
# 3m15 - Load data
# 6m20 - Test loading working
# 8m40 - NN class
# 10m10 - Multilayer NN Activation function
# 12m - Loss & optimizer
# 12m40 - Training loop
#
#
#
#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n device \n{ device }")

# Hyper-parameters - 1m30
input_size = 784            # images 28x28
hidden_size = 100 # 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Dataloader, Transformation
# MNIST dataset - download data unless local copy already present
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()
print(f"\n example_data \n{ example_data }")
#  example_data - guessing 0 is black!
# tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
#           [0., 0., 0.,  ..., 0., 0., 0.],
#           [0., 0., 0.,  ..., 0., 0., 0.],
#           ...,
#           [0., 0., 0.,  ..., 0., 0., 0.],
#           [0., 0., 0.,  ..., 0., 0., 0.],
#           [0., 0., 0.,  ..., 0., 0., 0.]]],

print(f"\n example_targets \n{ example_targets }")
#  example_targets
# tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5,
#         4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2,
#         4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 9, 3, 7, 4, 6, 4, 3, 0, 7, 0,
#         2, 9, 1, 7, 3, 2, 9, 7, 7, 6, 2, 7, 8, 4, 7, 3, 6, 1, 3, 6, 9, 3, 1, 4,
#         1, 7, 6, 9])

print(f"\n example_data.shape \n{ example_data.shape }")
#  example_data.shape
# torch.Size([100, 1, 28, 28])      # 100 images, 1 channel (no colour data), image 28x28

print(f"\n example_targets.shape \n{ example_targets.shape }")
#  example_targets.shape
# torch.Size([100])


# check data
import random
for i in range(12):
    plt.subplot(3,4,i+1)                            # 3 rows x 4 cols, index i+1
    img = random.randrange(0,100)
    plt.imshow(example_data[img][0], cmap='gray')
    # TODO add class label to sub plot
      #code

plt.show()


# 8m40 - NN class
class NeuralNet(nn.Module):
    '''
    input_size      tensor dimensions? or number of training elements 28x28 = 784
    hidden_size     number of hidden layers?
    num_classes     number of output classifications 0-9 10 in total
    '''
    # define model layers
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        # nn.Linear - layers
        # https://pytorch.org/docs/stable/nn.html#linear-layers
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        #
        # nn.ReLU - Non-linear Activations (weighted sum, nonlinearity)
        # https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU

    # 10m10 - Multilayer NN Activation function
    # forward pass functionality
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # softmax would be applied here but will be using Cross-Entropy - 10m45
        return out                                                    #
                                                                      #
# create model                                                        #
model = NeuralNet(input_size, hidden_size, num_classes)               #
                                                                      #
# 12m - Loss & Optimizer                                              #
criterion = nn.CrossEntropyLoss()   # < this applies softmax for us <<#
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 12m40 - Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')



#print(f"\n  \n{  }")





# Training Loop
# Model Evaluation
# GPU support



#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
