#! /usr/bin/env python

# PyTorch Tutorial 17 - Saving and Loading Models
# https://www.youtube.com/watch?v=9L9jEOwRrCg&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=18&t=25s

# to run
# cd /Users/simon/a_syllabus/lang/python/pytorch/pytorch_tut_00
# conda activate pt3
# ./scripts/17_tensor_save_load_models.py

#from __future__ import print_function
import torch
import torch.nn as nn
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)

# 0m - intro to gradients
# 1m - save / load complete model LAZY: torch.save(model, PATH) / model = torch.load(PATH)
# 1m50 - save/load RECOMMENDED: save model state dict
# 3m - EG code
# 3m30 - save load lazy way
# 5m40 - save load recommended way
# 9m - print / display state_dict
# 9m30 - common way of saving a check point - model building progress checkpoint I'm guessing
# 15m30 - Consideration when using GPU
# 15m50 - save on GPU, load on CPU - map_location
# 16m30 - save on GPU, load on GPU -
# 16m50 - save on CPU, load on GPU - map_location="cuda:0" select GPU


# 2m - computational graph
# etc

'''basic API
torch.save(arg, PATH)           # tensors, models, or any dict  - uses pickle under the hood
torch.load(PATH)

state_dict = model.state_dict()     # retrieve model parameters as dict - FROM model
model.load_state_dict(state_dict)   # restore model parameters as dict  - TO model
                                    # torch.nn.Module().load_state_dict()


# 1m - save / load complete model: torch.save(model, PATH) / model = torch.load(PATH)
# COMPLETE MODEL - LAZY WAY #
torch.save(model, PATH)
model = torch.load(PATH)
model.eval()                    # Lazy option - serialised data bound to specific classes
                                # and exact directory structure thats used when model saved

# 1m50 - save/load RECOMMENDED: save model state disct
# torch.save/load saves or loads a dict - model.state_dict()
# model.state_dict()         # retrieve model parameter
# model.load_state_dict()    # write model parameters
torch.save(model.state_dict(), PATH)    # model.state_dict() < model parameters

# recreate model using parameters
model = Model(*args, **kwargs)          # create model object
model.load_state_dict(torch.load(PATH)) # load dict - torch.load(PATH) pass it to load_state_dict()
model.eval()
'''

# 3m - EG code

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

# to run script
# cd /Users/simon/a_syllabus/lang/python/pytorch/pytorch_tut_00
# conda activate pt3
# ./scripts/17_tensor_save_load_models.py

# saving the LAZY way
FILE = "./data/model_whole.pth"             # pth extension is convention stands pytorch
torch.save(model, FILE)                     # lazy way

loaded_model = torch.load(FILE)
loaded_model.eval()

# see what got loaded
print(' Loop through parameters')
for param in loaded_model.parameters():
    print(param)


# saving the RECOMMENDED way - stave state dict
FILE_SD = "./data/model_state_dict.pth"
torch.save(model.state_dict(), FILE_SD)
print(f"\n model.state_dict() \n{ model.state_dict() }")

loaded_model = Model(n_input_features=6)        # create model object - n_input_features=6 ???

loaded_model.load_state_dict(torch.load(FILE_SD))  # load dict - torch.load(FILE_SD), & restore it to model

loaded_model.eval() # call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.? see bottom

print(f"\n loaded_model.state_dict() \n{ loaded_model.state_dict() }")

# 9m30 - common way of saving a check point - model building progress checkpoint I'm guessing
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f"\n TO SAVE optimizer.state_dict() \n{ optimizer.state_dict() }")
#  TO SAVE optimizer.state_dict()
# {'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0, 1]}]}
#                    learning rate ^               ^                                                ^ momentum type?

checkpoint = {
    "epoch": 90,                                # place in training
    "model_state": model.state_dict(),          # model state
    "optim_state": optimizer.state_dict()       # optimiser state < ALSO REQUIRED
}

FILE_CHECKPOINT = "checkpoint.pth"
torch.save(checkpoint, FILE_CHECKPOINT)

model = Model(n_input_features=6)                       # create model object
optimizer = torch.optim.SGD(model.parameters(), lr=0)   # create optimizer object - correct learning rate loaded later

checkpoint = torch.load(FILE_CHECKPOINT)        # load checkpoint - DICT like one above

model.load_state_dict(checkpoint['model_state'])        # restore model state_dict

optimizer.load_state_dict(checkpoint['optim_state'])    # restore optimiser state_dict
print(f"\n LOADED optimizer.state_dict() \n{ optimizer.state_dict() }")
#  LOADED optimizer.state_dict()
# {'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [0, 1]}]}
#               NOTE learning rate ^      ^ no longer 0

epoch = checkpoint['epoch']

model.eval()  # call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
# - or -
# model.train()

print(f"\n LOADED model.state_dict() \n{ model.state_dict() }")
# Remember that you must call model.eval() to set dropout and batch normalization layers
# to evaluation mode before running inference. Failing to do this will yield
# inconsistent inference results. If you wish to resuming training,
# call model.train() to ensure these layers are in training mode.



# 15m30 - Consideration when using GPU
""" SAVING ON GPU/CPU
# 15m50 - save on GPU, load on CPU - map_location
# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 16m30 - save on GPU, load on GPU
# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Note: Be sure to use the .to(torch.device('cuda')) function
# on all model inputs, too!

# 16m50 - save on CPU, load on GPU - map_location="cuda:0" select GPU
# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)
device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# This loads the model to a given GPU device.
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
"""


#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
