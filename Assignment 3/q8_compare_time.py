import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time


np.random.seed(42)


N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


# N_time_list_Normal = []
# N_time_list_grad_desc = []
# N_list = []
# P = 10
# for N in range(10,1000):
#     print(P,N)
#     N_list.append(N)
#     X = pd.DataFrame(np.random.randn(N, P))
#     y = pd.Series(np.random.randn(N))
    
#     ###################    Normal Equation

#     LR = LinearRegression(fit_intercept=True)
    
#     start = time.time()
#     LR.fit_normal(X, y)
#     end = time.time()
#     N_time_list_Normal.append((end-start)*(10**12))
#     ###################    Gradient Descent
    
#     start = time.time()
#     LR.fit_vectorised(X, y,len(X), n_iter=100, lr=0.01, lr_type='constant')
#     end = time.time()
#     N_time_list_grad_desc.append((end-start)*(10**12))


# # plt.title('Number of Features: 10, No of Examples : (30,10000)')
# plt.plot(N_list,N_time_list_Normal, label =  "Normal Equation")
# plt.plot(N_list,N_time_list_grad_desc, label = "Gradient Descent")

# plt.legend(loc='best')

# plt.show()

####################################### P-->varying
N_time_list_Normal = []
N_time_list_grad_desc = []
N_list = []

P = 10
for N in range(30,10000):
    print(P,N)
    N_list.append(N)
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    ###################    Normal Equation

    LR = LinearRegression(fit_intercept=True)
    
    start = time.time()
    LR.fit_normal(X, y)
    end = time.time()
    N_time_list_Normal.append((end-start))
    ###################    Gradient Descent
    
    start2 = time.time()
    LR.fit_vectorised(X, y,len(X), n_iter=30, lr=0.01, lr_type='constant')
    end2 = time.time()
    N_time_list_grad_desc.append((end2-start2))


print(N_time_list_Normal,N_time_list_grad_desc )
plt.title('Number of Features: 10, No of Examples : (30,10000)')
plt.plot(N_list,N_time_list_Normal, label =  "Normal Equation")
plt.plot(N_list,N_time_list_grad_desc, label = "Gradient Descent")

plt.legend(loc='best')

plt.show()