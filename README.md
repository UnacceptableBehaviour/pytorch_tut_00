# Introduction to Pytorch
REPO: [pytorch_tut_00](https://github.com/UnacceptableBehaviour/pytorch_tut_00)  
See [References](#references) for links to course content  


## Abstract
Work notes from Introduction to [Pytorch Beginner](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4).  

## Progress
KEY: (:white_check_mark:) watched, (:mag:) rewatch, (:flashlight:) unseen / to watch, (:question:) problem / open question  
CODE: (:seedling:) code complete, (:cactus:) incomplete / needs work, (:lemon:) not happy / code smells,  

Add table later if relevant.  


## Contents  
1. [Abstract](#abstract)  
2. [Progress](#progress)  
3. [Contents](#contents)  
4. [AIM:](#aim)  
5. [PyTorch Tutorials](#pytorch-tutorials)  
	1. [01 - Installation](#01---installation)  
	2. [02 - Tensor Basics](#02---tensor-basics)  
	3. [03 - Gradient Calculation With Autograd](#03---gradient-calculation-with-autograd)  
	4. [04 - Backpropagation - Theory With Example](#04---backpropagation---theory-with-example)  
	5. [05 - Gradient Descent with Autograd and Backpropagation](#05---gradient-descent-with-autograd-and-backpropagation)  
	6. [06 - Training Pipeline: Model, Loss, and Optimizer](#06---training-pipeline-model-loss-and-optimizer)  
	7. [07 - Linear Regression](#07---linear-regression)  
	8. [08 - Logistic Regression](#08---logistic-regression)  
	9. [09 - Dataset and DataLoader - Batch Training](#09---dataset-and-dataloader---batch-training)  
	10. [10 - Dataset Transforms](#10---dataset-transforms)  
	11. [11 - Softmax and Cross Entropy](#11---softmax-and-cross-entropy)  
	12. [12 - Activation Functions](#12---activation-functions)  
	13. [13 - Feed-Forward Neural Network](#13---feed-forward-neural-network)  
	14. [14 - Convolutional Neural Network (CNN)](#14---convolutional-neural-network-cnn)  
	15. [15 - Transfer Learning](#15---transfer-learning)  
	16. [16 - How To Use The TensorBoard](#16---how-to-use-the-tensorboard)  
	17. [17 - Saving and Loading Models](#17---saving-and-loading-models)  
	18. [18 - Create & Deploy A Deep Learning App - PyTorch Model Deployment With Flask & Heroku](#18---create--deploy-a-deep-learning-app---pytorch-model-deployment-with-flask--heroku)  
	19. [19 - PyTorch RNN Tutorial - Name Classification Using A Recurrent Neural Net](#19---pytorch-rnn-tutorial---name-classification-using-a-recurrent-neural-net)  
	20. [20 - RNN & LSTM & GRU - Recurrent Neural Nets](#20---rnn--lstm--gru---recurrent-neural-nets)  
	21. [21 - PyTorch Lightning Tutorial - Lightweight PyTorch Wrapper For ML Researchers](#21---pytorch-lightning-tutorial---lightweight-pytorch-wrapper-for-ml-researchers)  
	22. [22 - PyTorch LR Scheduler - Adjust The Learning Rate For Better Results](#22---pytorch-lr-scheduler---adjust-the-learning-rate-for-better-results)  
6. [EG chapter](#eg-chapter)  
	1. [EG episode](#eg-episode)  
		1. [**Vid contents - EG**](#vid-contents---eg)  
7. [Glossary of terms](#glossary-of-terms)  
8. [How To s](#how-to-s)  
	1. [How to install conda?](#how-to-install-conda)  
		1. [Whats conda?](#whats-conda)  
			1. [TLDR;](#tldr)  
	2. [How to install pytorch on osx?](#how-to-install-pytorch-on-osx)  
	3. [How do I autogenerate README.md file from RTF?](#how-do-i-autogenerate-readmemd-file-from-rtf)  
	4. [Where does conda store virtual environments?](#where-does-conda-store-virtual-environments)  
	5. [What is a conda channel?](#what-is-a-conda-channel)  
	6. [How do I install a conda environment?](#how-do-i-install-a-conda-environment)  
9. [References](#references)  


## AIM:  

Quick look a pytorch  dip toe in water!   



## PyTorch Tutorials  
### 01 - Installation 
([vid](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4))  
First install conda (s/w package manger similar to homebrew) [How to install conda](#how-to-install-conda)  
Then creat & activate the virtual environment and install pytorch [How to install pytorch on osx](#how-to-install-pytorch-on-osx)  
  
```
> conda create -n pt3 python=3.7        # -n pt3 - name of the virtual environment can be anything!
> conda activate pt3                    # activate the venv - # To deactivate use $ conda deactivate
> conda install pytorch torchvision torchaudio -c pytorch
```
WARNING: **do NOT set python=3.9** because the instal fails!  
Also  a bit of a gotcha:
```
(base) > conda activate pt3
(pt3) > python --version	             # using SYSTEM version
Python 3.9.2
(pt3) > python3 --version	             # using venv version 
Python 3.7.10
> python -c import sys; print(sys.executable)  # find out which exe is being used - rtf 
```
  
Conda [CHEAT SHEET](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)  
https://github.com/conda/conda/issues/9392  



### 02 - Tensor Basics  
### 03 - Gradient Calculation With Autograd  
### 04 - Backpropagation - Theory With Example  
### 05 - Gradient Descent with Autograd and Backpropagation  
### 06 - Training Pipeline: Model, Loss, and Optimizer  
### 07 - Linear Regression  
### 08 - Logistic Regression  
### 09 - Dataset and DataLoader - Batch Training  
### 10 - Dataset Transforms  
### 11 - Softmax and Cross Entropy  
### 12 - Activation Functions  
### 13 - Feed-Forward Neural Network  
### 14 - Convolutional Neural Network (CNN)  
### 15 - Transfer Learning  
### 16 - How To Use The TensorBoard  
### 17 - Saving and Loading Models  
### 18 - Create & Deploy A Deep Learning App - PyTorch Model Deployment With Flask & Heroku  
	[Docker Introduction 1hr](https://www.youtube.com/watch?v=i7ABlHngi1Q).  
### 19 - PyTorch RNN Tutorial - Name Classification Using A Recurrent Neural Net  
### 20 - RNN & LSTM & GRU - Recurrent Neural Nets  
### 21 - PyTorch Lightning Tutorial - Lightweight PyTorch Wrapper For ML Researchers  
### 22 - PyTorch LR Scheduler - Adjust The Learning Rate For Better Results  

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## EG chapter
### EG episode
[vid](https://www.youtube.com/watch?v=HtSuA80QTyo&list=PLUl4u3cNGP61Oq3tWYp6V_F-5jb5L2iHb&index=2&t=423s) ~ 
[lect notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/lecture-videos/MIT6_006F11_lec01.pdf)  

#### **Vid contents - EG**
 time				| notes	
| - | - |
**0m - 15m45**		| into to pytorch
**15m45 - 36m20**	|  installing.  

Maths equation test:
<p align="center"><img src="./tex/b01132dc54e41412aacb955f99104fe7.svg?invert_in_darkmode" align=middle width=679.0714507499999pt height=149.36606024999998pt/></p>

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  




## Glossary of terms




Note for array of **any** size tree: element A[n/2+1 . . n] are ALL leaves! 


## How To s
### How to install conda?
DLOAD miniconda installer [from here](https://conda.io/miniconda.html) - used python 3.9 bash.  
```
> shasum -a 256 /Users/simon/Downloads/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh  
b3bf77cbb81ee235ec6858146a2a84d20f8ecdeb614678030c39baacb5acbed1  /Users/simon/Downloads/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh  
SHA256 hash from dload link.  
b3bf77cbb81ee235ec6858146a2a84d20f8ecdeb614678030c39baacb5acbed1.  
Match!  
```
Install:
```
> bash /Users/simon/Downloads/Miniconda3-py39_4.9.2-MacOSX-x86_64.sh
Accept licence
Miniconda3 will now be installed into this location:
/Users/simon/miniconda3
Say yes to initialise, start new shell
> conda --version
conda 4.9.2
> conda update -n base -c defaults conda          # update to latest version
> conda create -n my_virtual_env_name python=3.9  # -n short name of the virtual environment can be anything!
> conda info --envs	                               # conda environments
# conda environments:
#
base                     /Users/simon/miniconda3
pt3                   *  /Users/simon/miniconda3/envs/pt
> conda remove --name env_name --all              # remove environment
```
[Setting up Virtual environments - basics](https://heartbeat.fritz.ai/creating-python-virtual-environments-with-conda-why-and-how-180ebd02d1db).  
[Conda Environments Python / R - TDS - more in depth](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533).  
  
#### Whats conda?  
[Difference between Conda & Pip?](https://www.anaconda.com/blog/understanding-conda-and-pip#:~:text=Pip%20installs%20Python%20packages%20whereas,software%20written%20in%20any%20language.&text=Another%20key%20difference%20between%20the,the%20packages%20installed%20in%20them.)  

##### TLDR;   
Conda more like Homebrew but it's cross platform
[more here](https://towardsdatascience.com/managing-project-specific-environments-with-conda-b8b50aa8be0e) 
inc Conda vs MiniConda vs Anaconda.  
  
Pip is the Python Packaging Authoritys recommended tool for installing packages from the Python Package Index, PyPI. Pip installs Python software packaged as wheels or source distributions. The latter may require that the system have compatible compilers, and possibly libraries, installed before invoking pip to succeed.  
  
Conda is a cross platform package and environment manager that installs and manages conda packages from the Anaconda repository as well as from the Anaconda Cloud. Conda packages are binaries. There is never a need to have compilers available to install them. Additionally conda packages are not limited to Python software. They may also contain C or C++ libraries, R packages or any other software.  
  

### How to install pytorch on osx?
[Pytorch Home Page](https://pytorch.org/).  
![install config](https://github.com/UnacceptableBehaviour/pytorch_tut_00/blob/main/imgs/pythorch_install_config.png)
```
> conda create -n pt3 python=3.7        # -n pt3 - name of the virtual environment can be anything!
> conda activate pt3                    # activate the venv - # To deactivate use $ conda deactivate
> conda install pytorch torchvision torchaudio -c pytorch
```
WARNING: **do NOT set python=3.9** because the instal fails!
  

### How do I autogenerate README.md file from RTF?
Use **create_TOC_for_md.py** which reads a specified RTF file & creates a TOC for the README.md file and the readme itself. 
It will also render & insert latex equations into the readme if needed.  
  
Requires striprtf for TOC function.
An adapted version of render int the same directory as create_TOC_for_md.py for the latex function - [avalable here](https://github.com/UnacceptableBehaviour/algorithms/blob/master/render.py).  
```
                                  # update for conda
> conda activate pt3
> pip install striprtf            # using **pip not RECOMMENDED** investigate options
> curl https://raw.githubusercontent.com/UnacceptableBehaviour/algorithms/master/create_TOC_for_md.py > create_TOC_for_md.py
> chmod +x create_TOC_for_md.py
# to pull render.py - if you need latex
> curl https://raw.githubusercontent.com/UnacceptableBehaviour/algorithms/master/render.py > render.py

# edit create_TOC_for_md.py point DEFAULT_DOC_TO_PROCESS=rtf source file
> mkdir -p ./scratch/tex          # -p make parent dirs as needed
                                  # add scratch dir to .gitignore - temp work area

> ./create_TOC_for_md.py -p       # to render README.md w/ TOC and maths equations


                                  # IF USING PIP
> .pe                             # alias .pe='. venv/bin/activate'
> pip install striprtf
> copy 
> ./create_TOC_for_md.py -p       # takes MATHS_00_MIT_6.042.rtf course notes and add TOC > README.md
                                  # also add README.md to git, commits, and pushes
                                  # -p = commit & push

> % conda list                    # list installed package showing install source (aka channel)
# packages in environment at /Users/simon/miniconda3/envs/pt:
#
# Name                    Version                   Build  Channel
openssl                   1.1.1k               h9ed2024_0  
.  
sqlite                    3.35.4               hce871da_0  
striprtf                  0.0.12                   pypi_0    pypi
etc
```

### Where does conda store virtual environments?
```
# with venv the path to venv is specified like so
> python3 -m venv /path/to/new/environment          
         
With conda theyre all in /Users/username/miniconda3/envs    # EG /Users/simon/miniconda3/envs/pt 
                                   Or .../anaconda3/envs
```
You ca also store them in the same way as venv see 
[Defnitive guide to conda](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533)  
see I prefer the approach taken by venv for two reasons.
 
### What is a conda channel?
Conda package sources are called channels, EG default_channels, conda-forge, pypi

### How do I install a conda environment?
Using a yml config file like so.
```
conda env create -f environment.yml          # as in pip install requirements.txt
                                             # require you to manage dependancies
conda search package_name --info
```

[Defnitive guide to conda](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533)   
[Pip & Conda](https://www.anaconda.com/blog/using-pip-in-a-conda-environment)   
Advice is dont mix them use conda.
If not available on conda option to [build conda packages](https://docs.conda.io/projects/conda-build/en/latest/) is available.
[Understanding Conda & Pip](https://www.anaconda.com/blog/understanding-conda-and-pip)   


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## References
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Courses / Books Found w/ Summary:








 