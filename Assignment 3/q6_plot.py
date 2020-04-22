import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression



x = np.array([i*np.pi/180 for i in range(60,1000,1)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

thetas = [[],[],[],[],[]]
degrees = [1,3,5,7,9]
N_s = [[],[],[],[],[]]
for degree in range(len(degrees)):
    for N in range(10,100):
        print(N)
        N_s[degree].append(N)
        x = np.array([i*np.pi/180 for i in range(N,300,4)])
        np.random.seed(10)  #Setting seed for reproducibility
        y = 4*x + 7 + np.random.normal(0,3,len(x))

        poly = PolynomialFeatures(degrees[degree])
        X_temp = poly.transform(x)
        X_temp = pd.DataFrame(X_temp)
        y = pd.Series(y)
        # print(X_temp)
        LR = LinearRegression(fit_intercept=False)
        # thetas_temp = LR.fit_vectorised(X_temp, y,30, n_iter=3, lr=0.00001, lr_type='constant')
        thetas_temp = LR.fit_normal(X_temp, y)

        # print(thetas_temp)
        thetas[degree].append(np.linalg.norm(thetas_temp))

    
for i in range(len(degrees)):
    plt.plot(N_s[i],thetas[i],label = "degree " + str(degrees[i]))
plt.legend(loc = "best")
plt.show()