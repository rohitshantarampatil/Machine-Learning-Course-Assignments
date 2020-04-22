import pandas as pd
import numpy as np
from numpy.linalg import inv
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt; plt.rcdefaults()
import math
class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''

        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        
        self.fit_intercept = fit_intercept
        self.param = None
        self.y_hat =None
        self.y = None
        self.X = None

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''

        self.X = X
        self.y = y

        #print("X.shape",X.shape)
        #print("y_shape",y.shape)
    
        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
        #print(X.head())
        if self.fit_intercept:
            X = pd.concat([bias,X],axis=1)
        #print(X.head())
        X = X.to_numpy()
        y = y.to_numpy()
        
        self.theta = np.dot(np.dot(inv(np.dot(X.transpose(),X)),X.transpose()),y)
        # print(self.theta)
    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
        if self.fit_intercept:
            X = pd.concat([bias,X],axis=1)
        X = X.to_numpy()
        y_hat = np.dot(X,self.theta)
        self.y_hat = y_hat
        return y_hat

    def plot(self):
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)

        """
        gs = gridspec.GridSpec(1, 3)

        plt.figure()
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        plt.scatter(self.y,self.y_hat)


        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        residual = []
        for i in range(len(self.y_hat)):
            residual.append(abs(self.y_hat[i]-self.y[i]))
        residual = pd.DataFrame(residual)
        residual.plot.kde()
        plt.xlabel("residuals")
        plt.ylabel("Probability density")
        title ="Mean : " +str(float(residual.mean()))+ " Variance : " +str(float(residual.var()))
        plt.title(title)
        plt.show()
        # plt.plot([0,1])
        ax = plt.subplot(gs[0,2]) # row 1, span all columns
        # plt.plot([0,1])
        objects = []
        if self.fit_intercept==True:
            objects.append(1)
        for i in self.X:
            objects.append(i)
        y_pos = np.arange(len(objects))
        performance = []
        for i in self.theta:
            performance.append(i)
        
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Coefficients')
        plt.title('Features and Coefficients')
        plt.show()