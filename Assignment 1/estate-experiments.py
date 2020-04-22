from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)


df = pd.read_excel('realestate.xlsx')
df = df.drop('No',axis=1)
attb = list(df.columns)
attb.remove('y')
#print(attb)

Dtree =  tree.DecisionTreeRegressor()
columns = list(df.columns)
columns.remove('y')
X = df[columns]
y = df['y']
y = y.rename('y')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)
#print(X_test.head())
#print(X_train, X_test, y_train, y_test)
print("Scikit-learn decision tree")
Dtree.fit(X_train,y_train)
pred = Dtree.predict(X_test)
print("MAE and STDDEV are ",mean_absolute_error(pred, y_test), np.std(np.abs(pred-y_test)))


print("Our Decision Tree")
tree = DecisionTree(criterion="a",max_depth=5)
tree.fit(X_train,y_train)
y_hat = tree.predict(X_test)
print("RMSE",rmse(y_hat,y_test))
print("MAE",mae(y_hat,y_test))

