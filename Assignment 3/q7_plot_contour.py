# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from linearRegression.linearRegression import LinearRegression
# import time


# ####################################################################
#                    ######PLOT - 2 #######
X = []
y = []
for i in range(15):
    X.append(i)
    y.append(3*i)
for i in range(0,15,2):
    y[i] = y[i]+1
for i in range(1,15,3):
    y[i] = y[i]-2
# plt.show()
temp_val = [-2,20]
X_temp = pd.DataFrame(temp_val)
X_n = pd.DataFrame(X)
y_n = pd.Series(y)
for i in range(100):
    LR = LinearRegression(fit_intercept=True)
    LR.fit_non_vectorised(X_n, y_n,15, n_iter=i, lr=0.0001, lr_type='constant')
    y_temp = LR.predict(X_temp)
    fig = plt.figure()
    ax = plt.subplot(111)    
    plt.plot(X_temp,y_temp,color = 'green', linewidth = '0.8')
    plt.xlim(-2,20)
    plt.ylim(-2,50)
    plt.scatter(X,y)
    # plt.ion()
    # plt.show()
    fig.savefig('plot2/plot'+str(i)+'.png')
#################################################################### 

# import numpy as np
# import matplotlib.pyplot as plt
# from preprocessing.polynomial_features import PolynomialFeatures
# import pandas as pd
# from linearRegression.linearRegression import LinearRegression


# x = np.array([i*np.pi/180 for i in range(60,300,4)])
# np.random.seed(10)  #Setting seed for reproducibility
# y = 4*x + 7 + np.random.normal(0,3,len(x))


# X_temp = x
# X_temp = pd.DataFrame(X_temp)
# y = pd.Series(y)
# print(X_temp)
# LR = LinearRegression(fit_intercept=True)
# LR.fit_vectorised(X_temp, y,30, n_iter=3, lr=0.001, lr_type='constant')
# thetas = LR.theta_list
# print(thetas)


#################  REFERENCE : https://xavierbourretsicotte.github.io/animation_ridge.html