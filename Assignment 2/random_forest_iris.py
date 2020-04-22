import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

dataset = load_iris()
X, y = dataset.data, dataset.target 
X = pd.DataFrame(X)
# y = pd.Series(y)
X['y'] = y
df = X.sample(frac=1)
df.reset_index(drop=True,inplace=True)
y = df.pop('y')
X = df.drop(columns=[0,2])
X = X.rename({1:0,3:1},axis=1)      
criteria = 'information_gain'
Classifier_AB = RandomForestClassifier(6, criterion = criteria )
Classifier_AB.fit(X[0:90], y[0:90])
y_hat = Classifier_AB.predict(X[90:150])
print(accuracy(y_hat,y[90:150]))
# Classifier_AB.plot()
for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat, y[90:150], cls))
    print('Recall for ',cls ,': ', recall(y_hat, y[90:150], cls))
