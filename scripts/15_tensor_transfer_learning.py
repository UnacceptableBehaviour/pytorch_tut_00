#! /usr/bin/env python

# PyTorch Tutorial 15 - Transfer Learning - using ResNet-18
# https://www.youtube.com/watch?v=K0lWSB2QoIQ&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=15
# Pytorch
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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

# https://www.youtube.com/watch?v=t6oHGXt04ik

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - Transfer learning - tweaking last layers of a neural net to classify a new target (similar data type)
# 2m - Resnet - 1M images, 18 layers, 1000 object categories,
# 2m20 - Topics: Image folder, Scheduler, Transfer Learning
# 3m46 - Image folder structure, training & validation folders
# 3m50 - Download Data from https://download.pytorch.org/tutorial/hymenoptera_data.zip
#   Scheduler
#
# 4m40 - Transfer Learning
#
# 5m30 - import pre-trained model- from torchvision import datasets, MODELS
#
#
#
#
#
#
#
# to run code
# cd /a_syllabus/lang/python/pytorch/pytorch_tut_00
# conda activate pt3        # name of the conda virtual envoronment
# ./scripts/15_tensor_transfer_learning.py


import torch                                # main framework
import torch.nn as nn                       # models
import torch.optim as optim                 # implementing various optimization algorithms.
    # Most commonly used methods are already supported
    # Note: if using GPU move to GPU before constructing optimizers
    # https://pytorch.org/docs/stable/optim.html
from torch.optim import lr_scheduler        # torch.optim.lr_scheduler provides several methods to adjust
    # the LEARNING RATE based on the number of EPOCHS.
    # torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based
    # on some validation measurements.
    #
    # Learning rate SCHEDULING SHOULD BE APPLIED AFTER OPTIMIZER’S update;
    # e.g., you should write your code this way:
    #
    # scheduler = ...
    # for epoch in range(100):
    #     train(...)
    #     validate(...)
    #     scheduler.step()
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# keys = ['k','l','m','n']
# cnt = 0
# def counter(k):
#     global cnt
#     cnt +=1
#     return f"{k}-{cnt}"
#
# syntax_test = { gen_key: counter(gen_key) for gen_key in keys }  # list comprehension for dict!
# print(f"\n syntax_test \n{ syntax_test }")
#
# dict_of_weapons = {'first': 'fear', 'second': 'surprise',
#                    'third':'ruthless efficiency', 'fourth':'fanatical devotion',
#                    'fifth': None}
#
# dict_comprehension = { k.upper(): weapon for k, weapon in dict_of_weapons.items() if weapon }
# print(f"\n dict_comprehension w/ condition \n{ dict_comprehension }")


data_dir = 'data/hymenoptera_data'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),        #
                                          data_transforms[x])
                  for x in ['train', 'val']}
print(f"\n type(image_datasets) \n{ type(image_datasets) }")
print(f"\n len(image_datasets) \n{ len(image_datasets) }")
print(f"\n image_datasets.keys() \n{ image_datasets.keys() }")
print(f"\n image_datasets \n{ image_datasets }")
#print(f"\n  \n{  }")
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
print(f"\n type(dataloaders) \n{ type(dataloaders) }")
print(f"\n len(dataloaders) \n{ len(dataloaders) }")


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(f"\n type(dataset_sizes) \n{ type(dataset_sizes) }")
print(f"\n len(dataset_sizes) \n{ len(dataset_sizes) }")

class_names = image_datasets['train'].classes
print(f"\n type(class_names) \n{ type(class_names) }")
print(f"\n len(class_names) \n{ len(class_names) }")
print(f"\n class_names \n{ class_names }")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n type(device) \n{ type(device) }")


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
#
#
# #### Finetuning the convnet ####
# # Load a pretrained model and reset final fully connected layer.
#
# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model.fc = nn.Linear(num_ftrs, 2)
#
# model = model.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that all parameters are being optimized
# optimizer = optim.SGD(model.parameters(), lr=0.001)
#
# # StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# # Decay LR by a factor of 0.1 every 7 epochs
# # Learning rate scheduling should be applied after optimizer’s update
# # e.g., you should write your code this way:
# # for epoch in range(100):
# #     train(...)
# #     validate(...)
# #     scheduler.step()
#
# step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#
# model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)
#
#
# #### ConvNet as fixed feature extractor ####
# # Here, we need to freeze all the network except the final layer.
# # We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
# model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False
#
# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 2)
#
# model_conv = model_conv.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#
# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=25)
#

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
