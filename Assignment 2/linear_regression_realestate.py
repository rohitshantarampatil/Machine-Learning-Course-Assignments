import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from metrics import *
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

df = pd.read_excel('realestate.xlsx')
df = df.drop('No',axis=1)
#print(df.head())


LR = LinearRegression(fit_intercept=True)
columns = list(df.columns)
columns.remove('y')
X = df[columns]
y = df['y']
y = y.rename('y')

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)
# X_train.reset_index(inplace = True, drop = True)
# X_test.reset_index(inplace = True,drop = True) 


# print("Linear Regression on real estate data")
#print(X_train.shape, y_train.shape)

# LR.fit(X_train,y_train)
# pred = LR.predict(X_test)
# print("MAE and STDDEV are ",mae(pred, y_test), np.std(np.abs(pred-y_test)))

################################## 5-fold cross validation #######################################

df = pd.read_excel('realestate.xlsx')
df = df.drop('No',axis=1)
#print(df.head())
columns = list(df.columns)

def cross_validtion_5_fold(X,y):
    X_original = X
    y_original = y
    MAE = []
    STDDEV = []
    for i in range(5):
        X_test = X.iloc[82*i:82*(i+1)][:]
        X_test.reset_index(drop=True,inplace=True)
        y_test = y[82*i:82*(i+1)]
        y_temp = pd.DataFrame(y)
        X_train = X.drop([j for j in range(82*i,82*(i+1))])
        X_train.reset_index(drop=True,inplace=True)
        y_train_temp = y_temp.drop([j for j in range(82*i,82*(i+1))])
        y_train_temp.reset_index(drop=True,inplace=True)
        y_train = None
        for j in y_train_temp:
            y_train = y_train_temp[j] 
        y_train.rename('y')
        LR.fit(X_train,y_train)
        pred = LR.predict(X_test)
        
        print("MAE and STDDEV for ",i+1,"th fold"," are ",mae(pred, y_test), np.std(np.abs(pred-y_test)))     
        MAE.append(mae(pred, y_test))
        STDDEV.append(np.std(np.abs(pred-y_test)))

    
    print("Average MAE and STDDEV are ",sum(MAE)/len(MAE), sum(STDDEV)/len(STDDEV))

cross_validtion_5_fold(X,y)