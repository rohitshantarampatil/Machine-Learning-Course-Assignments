
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    

    LR.fit_non_vectorised(X, y,30, n_iter=100, lr=0.01, lr_type='constant')
    y_hat = LR.predict(X)
    print('RMSE: for non vectorised with intercept = '+str(fit_intercept), rmse(y_hat, y))
    print('MAE:  RMSE: for non vectorised with intercept = '+str(fit_intercept), mae(y_hat, y))

    LR.fit_vectorised(X, y,30, n_iter=100, lr=0.01, lr_type='constant')
    y_hat = LR.predict(X)
    print('RMSE: for vectorised with intercept = '+str(fit_intercept), rmse(y_hat, y))
    print('MAE:  RMSE: for vectorised with intercept = '+str(fit_intercept), mae(y_hat, y))

    LR.fit_autograd(X, y,30, n_iter=100, lr=0.01, lr_type='constant') # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('RMSE: for autograd with intercept = '+str(fit_intercept), rmse(y_hat, y))
    print('MAE:  RMSE: for autograd with intercept = '+str(fit_intercept), mae(y_hat, y))
