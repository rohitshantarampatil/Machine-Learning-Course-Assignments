import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
import pandas as pd
from linearRegression.linearRegression import LinearRegression


x = np.array([i*np.pi/180 for i in range(60,300,4)])
x = x / x.max(axis=0)
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

thetas = []
degrees = []

for degree in range(1,10):
    degrees.append(degree)
    poly = PolynomialFeatures(degree)
    X_temp = poly.transform(x)
    X_temp = pd.DataFrame(X_temp)
    y = pd.Series(y)
    print(X_temp)
    LR = LinearRegression(fit_intercept=False)
    # thetas_temp = LR.fit_vectorised(X_temp, y,30, n_iter=50, lr=0.001, lr_type='constant')
    thetas_temp = LR.fit_normal(X_temp, y)

    # print(thetas_temp)
    thetas.append(np.linalg.norm(thetas_temp))

    
print(thetas)
print(degrees)
plt.title("q5 plot")
plt.xlabel('degree')
plt.ylabel('theta')
plt.plot(degrees,thetas)
plt.legend(loc = 'best')
plt.show()