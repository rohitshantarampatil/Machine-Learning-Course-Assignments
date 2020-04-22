"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
#from tree.base import DecisionTree
# Or use sklearn decision tree
from sklearn import tree
from linearRegression.linearRegression import LinearRegression

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 6
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
X_copy = X
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
# tree = DecisionTree(criterion=criteria)

Classifier_B = BaggingClassifier(base_estimator='tree', n_estimators=n_estimators )
Classifier_B.fit(X, y)
X_copy =  X_copy.drop(X_copy.columns[2],axis = 1)
y_hat = Classifier_B.predict(X_copy)
# Classifier_B.plot()
# [fig1, fig2] = Classifier_B.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat, y, cls))
    print('Recall for',cls,' : ', recall(y_hat, y, cls))


####################################  Reproducing Slides ###############

# X = []
# for i in range (1,9):
#     for j in range (1,9):
#         X.append([i,j])
# y = []
# for i in range (64):
#     y.append(0)

# l1 = [1,2,3,4,5,9,10,11,12,13,17,18,20,21,25,26,27,28,29,33,34,35,36,37,40]

# for i in l1:
#     y[i-1] = 1
# X = pd.DataFrame(X)
# y = pd.Series(y)
# X_copy = X
# Classifier_B = BaggingClassifier(base_estimator='tree', n_estimators=5 )
# Classifier_B.fit(X,y)
# X_copy =  X_copy.drop(X_copy.columns[2],axis = 1)

# y_hat = Classifier_B.predict(X_copy)
# Classifier_B.plot()
# [fig1, fig2] = Classifier_B.plot()
# criteria = 'information_gain'
# print('Criteria :', criteria)
# print('Accuracy: ', accuracy(y_hat, y))

# for cls in y.unique():
#     print('Precision: ', precision(y_hat, y, cls))
#     print('Recall: ', recall(y_hat, y, cls))