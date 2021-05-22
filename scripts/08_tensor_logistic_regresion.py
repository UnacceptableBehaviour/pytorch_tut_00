#! /usr/bin/env python

# PyTorch Tutorial 08 - Logistic Regression
# https://www.youtube.com/watch?v=OGpQxIkR4ao&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=8

#from __future__ import print_function
import torch
print("\n" * 20)
print("-" * 80)
print("-" * 80)
print("\n" * 2)


#### Steps in Torch ML pipeline
# 0) Prepare data set
# 1) Design Model (input, output size, forward pass)
# 2) Construct the loss & optimiser
# 3) Training Loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

# 0m - review Steps in Torch ML pipeline
# 1m - library imports
# 2m - Prepare data set - Breast Cancer
# 7m43 - Build model
# 14m40 - show accuracy


import torch

import torch.nn as nn       # PyTorch nn module has high-level APIs to build a neural network.
  # Torch. nn module uses Tensors and Automatic differentiation modules for training and building layers such as input,
  # hidden, and output layers - DOCS: https://pytorch.org/docs/stable/nn.html

import numpy as np      # NumPy is a library for the Python programming language, adding support for large,
  # multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate
  # on these arrays - DOCS: https://numpy.org/doc/stable/user/whatisnumpy.html

from sklearn import datasets  # to generate a regression dataset
                              # Scikit-learn is a library in Python that provides many unsupervised and supervised
  # learning algorithms. It contains a lot of efficient tools for machine learning and statistical modeling including
  # classification, regression, clustering and dimensionality reduction. Built upon some of the technology you might
  # already be familiar with, like NumPy, pandas, and Matplotlib!
  # DOCS: https://scikit-learn.org/stable/

from sklearn.preprocessing import StandardScaler        # to scale features NOTE NOT scalar! scaler for SCALING!

from sklearn.model_selection import train_test_split    # split training & test data


# 0) Prepare data set - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bc = datasets.load_breast_cancer()
#print(f"\n bc = datasets.load_breast_cancer() \n{ bc }")   # < SEE BOTTOM OF FILE
#print(bc.DESCR)                                            # < SEE BOTTOM OF FILE

X, y = bc.data, bc.target

n_samples, n_features = X.shape         # n_samples:569 n_features:30 < alot of feature he mentions
print(f"\n n_samples, n_features = X.shape \n n_samples:{n_samples} n_features:{ n_features }")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

# scale - recomended for logistic regression
sc = StandardScaler()                   # scaler used to scale data
X_train = sc.fit_transform(X_train)     # give features 0 mean, and some comment about variance
X_test = sc.transform(X_test)           # test data

# cast data - avoid problems later - COMMENT THIS OUT TO SEE
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# change shape fro row to column
#print(f"\n before y_train \n{ y_train }")                                                      # 1 ROW
#print(f"\n before y_train.shape[0] - y_train.shape \n{ y_train.shape[0] } - { y_train.shape }")# 455 - torch.Size([455])
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
#print(f"\n before y_train \n{ y_train }")
#print(f"\n after y_train.shape[0] - y_train.shape \n{ y_train.shape[0] } - { y_train.shape }") # 455 - torch.Size([455, 1])


# 1) Design Model (input, output size, forward pass) - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class LogisticRegression(nn.Module):
  def __init__(self, n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)    # input size n_input, output size = 1

  def forward(self, x):
    y_predicted = torch.sigmoid(self.linear(x))
    return y_predicted

# 30 input feature (n_features:30) & 1 output
model = LogisticRegression(n_features)


# 2) Construct the loss & optimiser - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
learning_rate = 0.01

criterion = nn.BCELoss()        # for LINEAR REGRESSION - BUILT IN Loss function - Binary Cross Entropy Loss
  # nn.BCELoss() creates a criterion - https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # SGD - Stocastic Gradient Descent
  # https://pytorch.org/docs/stable/optim.html?highlight=torch%20optim%20sgd#torch.optim.SGD
  # w/ optional Nesterov momentum :o

#print(f"\n  \n{  }")



# 3) Training Loop - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
num_epochs = 100

for epoch in range(num_epochs):
  #   - forward pass: compute prediction
  y_predicted = model(X_train)              # call model passing in data X
  loss = criterion(y_predicted, y_train)    # actual labels & predicted   - output = criterion(input, target)

  #   - backward pass: gradients
  loss.backward()

  #   - update weights
  optimizer.step()
  optimizer.zero_grad()

  if (epoch+1) % 10 == 0:
    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
  y_predicted = model(X_test)
  y_predicted_cls = y_predicted.round()
  acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
  print(f'accuracy = {acc:.4f}')


#print(f"\n  \n{  }")

#print(f"\n  \n{  }")
print('\n')


# bc = datasets.load_breast_cancer()
# print(f"\n bc = datasets.load_breast_cancer() \n{ bc }")
#
# bc = datasets.load_breast_cancer()
# {
# 'data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
#         1.189e-01],
#        [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
#         8.902e-02],
#        [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
#         8.758e-02],
#        ...,
#        [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
#         7.820e-02],
#        [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
#         1.240e-01],
#        [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
#         7.039e-02]]),
#  'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
#        0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
#        1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
#        1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
#        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
#        0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
#        1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
#        0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
#        1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
#        1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
#        0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
#        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
#        1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
#        1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
#        1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
#        1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
#        1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
#  'frame': None,
#  'target_names': array(['malignant', 'benign'], dtype='<U9'),
#  'DESCR': '.. _breast_cancer_dataset:\n\nBreast cancer wisc . . . 5177 char long . . SEE DESCR below.
#  'feature_names': array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#        'mean smoothness', 'mean compactness', 'mean concavity',
#        'mean concave points', 'mean symmetry', 'mean fractal dimension',
#        'radius error', 'texture error', 'perimeter error', 'area error',
#        'smoothness error', 'compactness error', 'concavity error',
#        'concave points error', 'symmetry error',
#        'fractal dimension error', 'worst radius', 'worst texture',
#        'worst perimeter', 'worst area', 'worst smoothness',
#        'worst compactness', 'worst concavity', 'worst concave points',
#        'worst symmetry', 'worst fractal dimension'], dtype='<U23'),
#  'filename': '/Users/simon/miniconda3/envs/pt3/lib/python3.7/site-packages/sklearn/datasets/data/breast_cancer.csv'
#  }

# 'DESCR': formated
#
# .. _breast_cancer_dataset:
#
# Breast cancer wisconsin (diagnostic) dataset
# --------------------------------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 569
#
#     :Number of Attributes: 30 numeric, predictive attributes and the class
#
#     :Attribute Information:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry
#         - fractal dimension ("coastline approximation" - 1)
#
#         The mean, standard error, and "worst" or largest (mean of the three
#         worst/largest values) of these features were computed for each image,
#         resulting in 30 features.  For instance, field 0 is Mean Radius, field
#         10 is Radius SE, field 20 is Worst Radius.
#
#         - class:
#                 - WDBC-Malignant
#                 - WDBC-Benign
#
#     :Summary Statistics:
#
#     ===================================== ====== ======
#                                            Min    Max
#     ===================================== ====== ======
#     radius (mean):                        6.981  28.11
#     texture (mean):                       9.71   39.28
#     perimeter (mean):                     43.79  188.5
#     area (mean):                          143.5  2501.0
#     smoothness (mean):                    0.053  0.163
#     compactness (mean):                   0.019  0.345
#     concavity (mean):                     0.0    0.427
#     concave points (mean):                0.0    0.201
#     symmetry (mean):                      0.106  0.304
#     fractal dimension (mean):             0.05   0.097
#     radius (standard error):              0.112  2.873
#     texture (standard error):             0.36   4.885
#     perimeter (standard error):           0.757  21.98
#     area (standard error):                6.802  542.2
#     smoothness (standard error):          0.002  0.031
#     compactness (standard error):         0.002  0.135
#     concavity (standard error):           0.0    0.396
#     concave points (standard error):      0.0    0.053
#     symmetry (standard error):            0.008  0.079
#     fractal dimension (standard error):   0.001  0.03
#     radius (worst):                       7.93   36.04
#     texture (worst):                      12.02  49.54
#     perimeter (worst):                    50.41  251.2
#     area (worst):                         185.2  4254.0
#     smoothness (worst):                   0.071  0.223
#     compactness (worst):                  0.027  1.058
#     concavity (worst):                    0.0    1.252
#     concave points (worst):               0.0    0.291
#     symmetry (worst):                     0.156  0.664
#     fractal dimension (worst):            0.055  0.208
#     ===================================== ====== ======
#
#     :Missing Attribute Values: None
#
#     :Class Distribution: 212 - Malignant, 357 - Benign
#
#     :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
#
#     :Donor: Nick Street
#
#     :Date: November, 1995
#
# This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
# https://goo.gl/U2Uwz2
#
# Features are computed from a digitized image of a fine needle
# aspirate (FNA) of a breast mass.  They describe
# characteristics of the cell nuclei present in the image.
#
# Separating plane described above was obtained using
# Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
# Construction Via Linear Programming." Proceedings of the 4th
# Midwest Artificial Intelligence and Cognitive Science Society,
# pp. 97-101, 1992], a classification method which uses linear
# programming to construct a decision tree.  Relevant features
# were selected using an exhaustive search in the space of 1-4
# features and 1-3 separating planes.
#
# The actual linear program used to obtain the separating plane
# in the 3-dimensional space is that described in:
# [K. P. Bennett and O. L. Mangasarian: "Robust Linear
# Programming Discrimination of Two Linearly Inseparable Sets",
# Optimization Methods and Software 1, 1992, 23-34].
#
# This database is also available through the UW CS ftp server:
#
# ftp ftp.cs.wisc.edu
# cd math-prog/cpo-dataset/machine-learn/WDBC/
#
# .. topic:: References
#
#    - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction
#      for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on
#      Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
#      San Jose, CA, 1993.
#    - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and
#      prognosis via linear programming. Operations Research, 43(4), pages 570-577,
#      July-August 1995.
#    - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
#      to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994)
#      163-171.

