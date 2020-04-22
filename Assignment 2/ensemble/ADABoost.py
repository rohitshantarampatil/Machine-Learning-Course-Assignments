from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
import math

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_list = []
        self.alphas = []
        self.classes =[]
        self.clfs = []
        self.clfsy=[]
        self.data = None
        self.labels = None
    def fit(self, X, y):
        self.data = X
        self.labels = y
        self.classes = list(set(y))
        n = len(X)
        weights = [1/n for i in range(n)]
        for estimator in range(self.n_estimators):
            self.clfs.append(X)
            self.clfsy.append(y)
            Dtree =  DecisionTree("information_gain",max_depth=1)
            Dtree.fit(X,y,sample_weights=weights)            
            self.estimators_list.append(Dtree)
            err = 0         
            for i in range(n):
                if Dtree.predict(X.iloc[[i]])!=y[i]:
                    err += weights[i]
            alpha = 0.5*math.log2((1-err)/err)
            self.alphas.append(alpha)
            for i in range(n):
                if Dtree.predict(X.iloc[[i]])!=y[i]:
                    weights[i]=weights[i]*math.exp(alpha)
                else:
                    weights[i]=weights[i]*math.exp(-alpha)
            #Normalise the weights
            temp = [t/sum(weights) for t in weights]
            weights = temp
    def predict(self, X):
        y_hat = np.zeros(len(X))
        maindict = {self.classes[0]:1,self.classes[1]:-1}
        for i in range(len(X)):
            tot_pred = 0
            for j in range(self.n_estimators):
                # print(int(self.estimators_list[j].predict(X.iloc[[i]])))
                tot_pred += self.alphas[j]*maindict[int(self.estimators_list[j].predict(X.iloc[[i]]))]
            if tot_pred>0:
                y_hat[i] = self.classes[0]
            else:
                y_hat[i] = self.classes[1]
        return y_hat        





    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        print("Printing decision surfaces of decision trees")
        plot_colors = "gb"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(1, 3, _+1 )
            x_min, x_max = self.clfs[_].iloc[:, 0].min() - 1, self.clfs[_].iloc[:, 0].max() + 1
            y_min, y_max = self.clfs[_].iloc[:, 1].min() - 1, self.clfs[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.estimators_list[_].predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.clfsy[_] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.clfs[_].loc[idx[0][i]][0], self.clfs[_].loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("Decision surface of a decision trees")
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
        Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(self.labels == i)
            for i in range (len(idx[0])):
                plt.scatter(self.data.loc[idx[0][i]][0], self.data.loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("Combined decision surface ")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig2 = plt

        return [fig1,fig2]

