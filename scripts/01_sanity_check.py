#! /usr/bin/env python

# > conda create -n pt3 python=3.7        # -n pt3 - name of the virtual environment can be anything!
# > conda activate pt3                    # activate the venv - # To deactivate use $ conda deactivate
# > conda install pytorch torchvision torchaudio -c pytorch

# WARNING: **do NOT set python=3.9** because the instal fails!

from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
