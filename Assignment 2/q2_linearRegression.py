import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from metrics import *
import math
from sklearn.model_selection import train_test_split


np.random.seed(42)


N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

def time_complexity():
    N_time_list = []
    N_list = []
    P = 10
    for N in range(30,10000):
        print(P,N)
        N_list.append(N)
        start = time.time()
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit(X, y)
        end = time.time()
        N_time_list.append((end-start)*(10**12))
    plt.title('Number of Features: 10, No of Examples : (30,10000)')
    plt.plot(N_list,N_time_list)
    plt.show()
    

    N_time_list = []
    N_list = []
    N = 100
    for P in range(50,1000):
        print(P,N)
        N_list.append(P)
        start = time.time()
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit(X, y)
        end = time.time()
        N_time_list.append((end-start)*(10**12))
    plt.title('Number of Examples:100, No of Features : (50,1000)')
    plt.plot(N_list,N_time_list)
    plt.show()




for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = LR.predict(X)
    LR.plot()
    # time_complexity()
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

