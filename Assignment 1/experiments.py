
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

# N = 30
# P = 5

############################################################################################################
#    									DISCRETE DESCRETE
fit_time = []
predict_time =[]
for N in range(2,10):
	for P in range(100,120):
		X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
		y = pd.Series(np.random.randint(P, size = N) , dtype="category")
		tree = DecisionTree(criterion="a",max_depth=8) #Split based on Inf. Gain
		start = time.time()
		tree.fit(X, y)
		end = time.time()
		fit_time.append(end-start)

		start = time.time()
		y_hat = tree.predict(X)
		end  = time.time()
		predict_time.append(end-start)
plt.plot(fit_time)
plt.ylabel('DIDO : Fit time', fontsize=16)
plt.show()

plt.plot(predict_time)
plt.ylabel('DIDO : Predict time', fontsize=16)
plt.show()
###########################################################################################################
# #										REAL DISCRETE											
fit_time = []
predict_time =[]
for P in range(2,10):
	for N in range(100,120):
		print(N,P)
		X = pd.DataFrame(np.random.randn(N, P))
		y = pd.Series(np.random.randint(P, size = N), dtype="category")
		tree = DecisionTree(criterion="a",max_depth=4) #Split based on Inf. Gain
		start = time.time()
		tree.fit(X, y)
		end = time.time()
		fit_time.append(end-start)

		start = time.time()
		y_hat = tree.predict(X)
		end  = time.time()
		predict_time.append(end-start)
plt.plot(fit_time)
#print(fit_time)
plt.ylabel('RIDO : Fit time', fontsize=16)
plt.show()

plt.plot(predict_time)
plt.ylabel('RIDO : Predict time', fontsize=16)
plt.show()
###############################################################################################################
#                                           DISCRETE REAL
fit_time = []
predict_time =[]
for N in range(2,10):
	for P in range(100,120):
		X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
		y = pd.Series(np.random.randn(N))
		tree = DecisionTree(criterion="a",max_depth=4) #Split based on Inf. Gain
		start = time.time()
		tree.fit(X, y)
		end = time.time()
		fit_time.append(end-start)

		start = time.time()
		y_hat = tree.predict(X)
		end  = time.time()
		predict_time.append(end-start)
plt.plot(fit_time)
plt.ylabel('DIRO : Fit time', fontsize=16)
plt.show()

plt.plot(predict_time)
plt.ylabel('DIRO : Predict time', fontsize=16)
plt.show()


##################################################################################################################
#                                      REAL REAL
fit_time = []
predict_time =[]
for P in range(2,10):
	for N in range(100,120):
		X = pd.DataFrame(np.random.randn(N, P))
		y = pd.Series(np.random.randn(N))
		tree = DecisionTree(criterion="a",max_depth=4) #Split based on Inf. Gain
		start = time.time()
		tree.fit(X, y)
		end = time.time()
		fit_time.append(end-start)

		start = time.time()
		y_hat = tree.predict(X)
		end  = time.time()
		predict_time.append(end-start)
plt.plot(fit_time)
plt.ylabel('RIRO : Fit time', fontsize=16)
plt.show()

plt.plot(predict_time)
plt.ylabel('RIRO : Predict time', fontsize=16)
plt.show()
