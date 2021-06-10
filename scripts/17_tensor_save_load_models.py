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

# 2m - computational graph
# etc

'''basic API
torch.save(arg, PATH)           # tensors, models, or any dict  - uses pickle under the hood
torch.load(PATH)

state_dict = model.state_dict()     # retrieve model parameters as dict - FROM model
model.load_state_dict(state_dict)   # restore model parameters as dict  - TO model


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
FILE = "./data/model_whole.pth"                   # pth extension is convention stands pytorch
torch.save(model, FILE)                     # lazy way

loaded_model = torch.load(FILE)
loaded_model.eval()

# see what got loaded
print(' Loop through parameters')
for param in loaded_model.parameters():
    print(param)


# saving the RECOMMENDED way - stave state dict
FILE = "model_state_dict.pth"
torch.save(model.state_dict(), FILE)
print(f"\n model.state_dict() \n{ model.state_dict() }")

loaded_model = Model(n_input_features=6)        # create model object - n_input_features=6 ???

loaded_model.load_state_dict(torch.load(FILE))  # load dict - torch.load(FILE), & restore it to model

loaded_model.eval()

print(f"\n loaded_model.state_dict() \n{ loaded_model.state_dict() }")

#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')
