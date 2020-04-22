from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
import math

def random_split(X_t,y_t):
    X_t['y'] = y_t
    lst = []
    for i in range(len(X_t)):
        sample = X_t.sample(n=1)
        lst.append(sample)
    X_t = pd.concat(lst)
    X_t.reset_index(drop=True,inplace = True)
    # X_t =X_t.sample(n=len(X_t)//2)
    # X_t.reset_index(drop=True,inplace = True)
    y_t = X_t.pop('y')
    return X_t,y_t
class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_list = []
        self.clfs = []
        self.clfsy = []
        self.data = None
        self.labels = None

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.data = X
        self.labels = y
        for estimator in range(self.n_estimators):
            Dtree =  tree.DecisionTreeClassifier(max_depth=4)
            X_train,y_train = random_split(X,y)
            self.clfs.append(X_train)
            self.clfsy.append(y_train)
            # print(X_train,y_train)
            Dtree.fit(X_train,y_train)
            self.estimators_list.append(Dtree)


    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = np.zeros(len(X))
        predictions = []
        for i in self.estimators_list:
            predictions.append(i.predict(X))
        pred_arr = np.array(predictions)
        pred_arr = pred_arr.T
        y_hat = [np.argmax(np.bincount(i)) for i in pred_arr]
        return(pd.Series(y_hat))

    def plot(self):
        print("Printing decision surfaces of decision trees")
        plot_colors = "gb"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(2, 3, _+1 )
            x_min, x_max = self.clfs[_].iloc[:, 0].min() - 1, self.clfs[_].iloc[:, 0].max() + 1
            y_min, y_max = self.clfs[_].iloc[:, 1].min() - 1, self.clfs[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.estimators_list[_].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.clfsy[_] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.clfs[_].loc[idx[0][i]][0], self.clfs[_].loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig1 = plt

        # Figure 2
        print("Printing decision surface by combining the individual estimators")
        plot_colors = "gb"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.data.iloc[:, 0].min() - 1, self.data.iloc[:, 0].max() + 1
        y_min, y_max = self.data.iloc[:, 1].min() - 1, self.data.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(self.labels == i)
            for i in range (len(idx[0])):
                plt.scatter(self.data.loc[idx[0][i]][0], self.data.loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("Decision surface by combining individual estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig2 = plt

        return [fig1,fig2]
 