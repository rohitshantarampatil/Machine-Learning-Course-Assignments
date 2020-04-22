"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class DecisionTree():
    def __init__(self,criterion,max_depth):
        self.criterion = criterion
        self.max_depth = max_depth
        self.feature= None
        self.splitval = None
        self.preds = None

    def fit(self, X, y,sample_weights):

        if self.max_depth == 1:
            attr_list = []
            igf = []
            splitval_list = []
            preds_list = []
            for i in X:
                inf_gain,splitval,pred_l,pred_g = information_gain(y,X[i],sample_weights)
                attr_list.append(i)
                igf.append(inf_gain)
                splitval_list.append(splitval)
                preds_list.append([pred_l,pred_g])
            self.feature = attr_list[igf.index(max(igf))]
            self.splitval = splitval_list[igf.index(max(igf))]
            self.preds = preds_list[igf.index(max(igf))]
    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        if self.max_depth == 1:
            y_hat = np.zeros(len(X))
            for i in range(len(X)):
                if X.iloc[i][self.feature]<=self.splitval:
                    val = self.preds[0]    
                else:
                    val = self.preds[1]
                y_hat[i] = val
            return y_hat
    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass

# N = 30
# P = 2
# NUM_OP_CLASSES = 2
# n_estimators = 3
# X = pd.DataFrame(np.abs(np.random.randn(N, P)))
# y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")
# n = len(X)
# weights = [1/n for i in range(n)]
# Dtree =  DecisionTree("inf_gain",max_depth=1)
# Dtree.fit(X,y,sample_weights=weights)
# y_hat = Dtree.predict(X)
# print(y_hat)
# print(y)