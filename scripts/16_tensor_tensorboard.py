#! /usr/bin/env python

# PyTorch Tutorial 03 - Gradient Calculation With Autograd
# https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - Intro to https://www.tensorflow.org/tensorboard
# 1m - Tools: Track & Vis metrics, model graphs, redimensioning, Profiling TensorFlow programs all sorts
# 1m10 - Code from 13 - Feed-Forward Neural Network - MNIST digit classi
# 2m30 - Install tensorboard - conda install -c conda-forge tensorboard in the vid he uses pip
# 3m40 - import tesnorboard, setup
# 4m30 - add images to tensorboard
# 6m20 - image view test
# 7m - Adding a graph
# 8m40 - Inspect graph Add to Ep 13,
# 9m - adding Accuracy & Loss - writer.add_scalar
# 14m30 - modifying the learning rate
# 16m - prescision & recal curve? https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# note - Wow! opened a can of terms leading to the confusion matrix!! [used t](https://en.wikipedia.org/wiki/Confusion_matrix)
# 16m20 - TensorBoard doc add_pr_curve
# 17m20 - add code to do PR curve for each classification
# 19n20 - import torch.nn.functional as F to convert last layer output into softmax probabilities
# 20m40 - Use class_probs_batch = [F.softmax(output, dim=0) for output in outputs] to convert values
#
#

# 16m - prescision & recal curve - aka PR curve https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# Wow! opened a can of terms leading to the confusion matrix!! [used t](https://en.wikipedia.org/wiki/Confusion_matrix)

# prescision & recal curve
# accuracy =  True positives / total observations (Note: imbalanced data problem will give misleading results)
# balanced data - classes have similar number of elements
# unbalanced date - classes have very different number of elements skewing accuracy
# decision threshold - classification boundary
# TP - True Positive
# FN - False Negative
# FP - False Positive
# AUC - Area under curve
# ROC - Receiver Operating Characteristic, for predicting the probability of a binary outcome
#
# precision y axis range 0-1    out of all the time I predicted a positive how many time was I correct
#                               total correct classifications / total observations in that class
#                               precision = TP / (TP + FP)
#
# recall x axis    range 0-1    total correct classifications / total observations in the whole model
#                               recall = TP / (TP + FN)
#
# best result is 1,1 < ideal

# tutorial w code https://www.youtube.com/watch?v=_UEBIOC4WIY
# tutorial w code https://www.youtube.com/watch?v=gZmOmgQns3c

#
# Whats a ROC curve? Receiver Operating Characteristic
# https://datascience103579984.wordpress.com/2019/04/30/roc-and-precision-recall-curves/
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# Terms
# SESNSITIVITY (True Positive Rate)
# vs
# SPECIFICITY (False Positive Rate)
#
#
# When PREVALENCE matters a PR curve is used instead of a ROC curve
# Prevalence = P / P + N
# What is PREVALENCE


# https://www.youtube.com/watch?v=_1QtMPuYIVw

# https://creativecommons.org/licenses/by-sa/4.0/

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist1')
###################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# MNIST dataset
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

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

############## TENSORBOARD ########################
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
#writer.close()
#sys.exit()
###################################################

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############## TENSORBOARD ########################
writer.add_graph(model, example_data.reshape(-1, 28*28))
#writer.close()
#sys.exit()
###################################################

# Train the model
running_loss = 0.0
running_correct = 0
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

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            ############## TENSORBOARD ########################
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0
            ###################################################

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
class_labels = []
class_preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        values, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_probs_batch)
        class_labels.append(predicted)

    # 10000, 10, and 10000, 1  < tensor shape
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    ############## TENSORBOARD ########################
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]     # all sample but only fro class i
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    ###################################################

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
