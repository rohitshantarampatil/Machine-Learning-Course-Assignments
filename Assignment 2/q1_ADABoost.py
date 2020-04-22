"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn import tree
from linearRegression.linearRegression import LinearRegression
from sklearn.datasets import load_iris

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
# tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator='tree', n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat, y, cls))
    print('Recall for ',cls ,': ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
dataset = load_iris()
X, y = dataset.data, dataset.target 
X = pd.DataFrame(X)
# y = pd.Series(y)
for i in range(len(y)):
    if y[i]==1:
        y[i] = 0
for i in range(len(y)):
    if y[i]==2:
        y[i] = 1   

X['y'] = y
df = X.sample(frac=1)
df.reset_index(drop=True,inplace=True)
y = df.pop('y')
X = df
X = df.drop(columns=[0,2])
X = X.rename({1:0,3:1},axis=1)             
criteria = 'information_gain'
Classifier_AB = AdaBoostClassifier(base_estimator='tree', n_estimators=3 )
Classifier_AB.fit(X[0:90], y[0:90])
y_hat = Classifier_AB.predict(X[90:150])
print(accuracy(y_hat,y[90:150]))
# Classifier_AB.plot()
for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat, y[90:150], cls))
    print('Recall for ',cls ,': ', recall(y_hat, y[90:150], cls))

###### For Comparison with decision stump

stump = tree.DecisionTreeClassifier()
stump.fit(X[0:90], y[0:90])
y_hat = stump.predict(X[90:150])
print(accuracy(y_hat,y[90:150]))