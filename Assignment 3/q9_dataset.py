import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

# x_1 = [ i for i in range(100)]
# x_2 = [ i for i in range(200,300)]
# x_3 = [i+j for i in x_1, j in x_2]
# print(len(x_3))

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = X[1]
X[5] = X[1]


LR = LinearRegression(fit_intercept=True)
thetas = LR.fit_vectorised(X, y,len(X), n_iter=1000, lr=0.01, lr_type='constant')
print(thetas)