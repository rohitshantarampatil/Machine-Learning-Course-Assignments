import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad
from numpy.linalg import inv,pinv

# Import Autograd modules here

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.thetas = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.theta_list = []
        self.x = None
    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        y =y.rename('y')
        self.num_examples,self.num_features = X.shape
        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))

        if self.fit_intercept:
            self.thetas = np.random.randn(self.num_features+1)
            X = pd.concat([bias,X],axis=1)
        else:
            self.thetas = np.random.randn(self.num_features)

        dataset = pd.concat([X,y],axis=1)
        self.alpha = lr
        for iteration in range(n_iter):
            if lr_type=='inverse':
                self.alpha = lr/(iteration+1)
            sample = dataset.sample(n=batch_size)
            y_s = sample.pop('y')
            X_s = sample
            num_examples = batch_size
            epsilon = np.dot(X_s,self.thetas)-y_s
            # epsilon = []
            cost = np.sum((epsilon)**2)/num_examples
            gradient = np.dot (X_s.transpose(),epsilon)/(num_examples*0.5)
            for i in range(len(self.thetas)):
                self.thetas[i] = self.thetas[i] - self.alpha*gradient[i]

        return self.thetas  
    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        y =y.rename('y')
        self.num_examples,self.num_features = X.shape
        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))

        if self.fit_intercept:
            self.thetas = np.zeros(self.num_features+1)
            X = pd.concat([bias,X],axis=1)
        else:
            self.thetas = np.random.randn(self.num_features)
        
        # self.thetas = np.asarray([-2,8])
        self.x = X
        dataset = pd.concat([X,y],axis=1)
        self.alpha = lr
        for iteration in range(n_iter):
            if lr_type=='inverse':
                self.alpha = lr/(iteration+1)
            sample = dataset.sample(n=batch_size)
            y_s = sample.pop('y')
            X_s = sample
            num_examples = batch_size
            epsilon = np.dot(X_s,self.thetas)-y_s
            cost = np.sum((epsilon)**2)/num_examples
            gradient = np.dot (X_s.transpose(),epsilon)/(num_examples*0.5)
            # print("gradient",gradient)
            self.thetas = self.thetas - self.alpha*gradient
            self.theta_list.append(self.thetas)

        return self.thetas  

            
    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        def cost_funct(thetas):
            epsilon = np.dot(X_s,thetas)-y_s
            return np.sum((epsilon)**2)/num_examples


        y =y.rename('y')
        self.num_examples,self.num_features = X.shape
        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))

        if self.fit_intercept:
            self.thetas = np.random.randn(self.num_features+1)
            X = pd.concat([bias,X],axis=1)
        else:
            self.thetas = np.random.randn(self.num_features)

        dataset = pd.concat([X,y],axis=1)
        self.alpha = lr
        gradient = grad(cost_funct)

        for iteration in range(n_iter):
            if lr_type=='inverse':
                self.alpha = lr/(iteration+1)
            sample = dataset.sample(n=batch_size)
            y_s = sample.pop('y')
            X_s = sample
            num_examples = batch_size
            epsilon = np.dot(X_s,self.thetas)-y_s
            # cost = np.sum((epsilon)**2)/num_examples
            # gradient = np.dot (X_s.transpose(),epsilon)/(num_examples)
            self.thetas = self.thetas - self.alpha*gradient(self.thetas)

        return self.thetas

        

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
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
        
        self.thetas = np.dot(np.dot(inv(np.dot(X.transpose(),X)),X.transpose()),y)
        return self.thetas
    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))

        if self.fit_intercept:
            X = pd.concat([bias,X],axis=1)

        return np.dot(X,self.thetas)

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
