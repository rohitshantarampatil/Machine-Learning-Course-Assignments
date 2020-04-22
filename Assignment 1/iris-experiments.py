import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

import sys
from sklearn.datasets import load_iris

np.random.seed(42)

# Read IRIS data set
# ...

# 
dataset = load_iris()
X, y = dataset.data, dataset.target 

#from sklearn.utils import shuffle
#X, y = shuffle(X, y, random_state=0)

print("fit model for iris dataset for 70-30 division")

clf = DecisionTree(criterion="a",max_depth=5)
clf.fit(pd.DataFrame(X[0:120]),pd.Series(y[0:120],dtype = "category"))
y = y[120:]
y_hat = clf.predict(pd.DataFrame(X[120:]))
print("Accuracy",accuracy(pd.Series(y_hat),pd.Series(y)))
y = pd.Series(y)

for cls in y.unique():
    print('Precision: for class ',cls," : ", precision(y_hat, y, cls))
    print('Recall: ',cls," : ", recall(y_hat, y, cls))


def cross_validtion_5_fold(X,y,depth):
    X_original = X
    y_original = y

    clf = DecisionTree(criterion="a",max_depth=depth)
    clf.fit(pd.DataFrame(X[0:120]),pd.Series(y[0:120],dtype = "category"))
    y = y[120:]
    y_hat = clf.predict(pd.DataFrame(X[120:]))
    print(accuracy(pd.Series(y_hat),pd.Series(y)))

    X= X_original
    y = y_original

    clf = DecisionTree(criterion="a",max_depth=depth)
    clf.fit( pd.DataFrame(np.append(X[90:],X[0:60],axis=0)), pd.Series(np.append(y[90:],y[0:60],axis=0),dtype = "category"))
    y = y[60:90]
    y_hat = clf.predict(X[60:90])
    print(accuracy(pd.Series(y_hat),pd.Series(y)))

    X= X_original
    y = y_original
    
    clf = DecisionTree(criterion="a",max_depth=depth)
    clf.fit(pd.DataFrame(np.append(X[120:],X[0:90],axis=0)), pd.Series(np.append(y[120:],y[0:90],axis=0),dtype="category"))
    y = y[90:120]
    y_hat = clf.predict(X[90:120])
    print(accuracy(pd.Series(y_hat),pd.Series(y)))

    X= X_original
    y = y_original
    
    clf = DecisionTree(criterion="a",max_depth=depth)
    clf.fit(pd.DataFrame(X[30:]), pd.Series(y[30:],dtype="category"))
    y = y[0:30]
    y_hat = clf.predict(X[0:30])
    print(accuracy(pd.Series(y_hat),pd.Series(y)))

    X= X_original
    y = y_original
    
    clf = DecisionTree(criterion="a",max_depth=depth)
    clf.fit(pd.DataFrame(np.append(X[0:30],X[60:],axis=0)), pd.Series(np.append(y[0:30],y[60:],axis=0),dtype="category"))
    y = y[30:60]
    y_hat = clf.predict(X[30:60])
    print(accuracy(pd.Series(y_hat),pd.Series(y)))



def nested_cross_validation(dataset,y):
    for i in range(5):
        test = dataset[30*i:30*(i+1)]
        test_label = y[30*i:30*(i+1)]
        if 30*(i+1)+120<=150:
            train = dataset[30*(i+1):]
            train_label = y[30*(i+1):]
            #print("yo")


        else:
            train1 = dataset[0:30*(i+1)-30]
            train1_label = y[0:30*(i+1)-30]
            train2 = dataset[30*(i+1):]
            train2_label = y[30*(i+1):]
            train = np.append(train1,train2,axis=0)
            train_label = np.append(train1_label,train2_label,axis=0)
            #print("yoo")
        accuracy_validation = {}
        for depth in range(1,11):
            avg_acc = 0
            for j in range(4):
                #print("yooooo")
                #print(train.shape,train_label.shape)
                validation = train[30*j:30*(j+1)]
                validation_label = train_label[30*j:30*(j+1)]
                train_1 = train[30*(j+1):]
                train1_label = train_label[30*(j+1):]
                train_2 = train[0:30*(j+1)-30]
                train2_label = train_label[0:30*(j+1)-30]
                train_new= np.append(train_1,train_2,axis = 0)
                train_new_label = np.append(train1_label,train2_label,axis=0)
                tree = DecisionTree(criterion="gini_index",max_depth=depth)
                #print(pd.DataFrame[train])
                #print(train_new.shape,train_new_label.shape)
                #print(train_new.shape,train_new_label.shape)
                train_new=pd.DataFrame(train_new)
                train_new_label = pd.Series(train_new_label,dtype="category")
                train_new.reset_index(drop=True,inplace= True)
                train_new_label.reset_index(drop=True,inplace= True)
                #print(train_new)
                #print(train_new_label)
                tree.fit(train_new,train_new_label)
                #print("training done")
                #print("now testing")
                avg_acc+= accuracy(tree.predict(validation),validation_label)
                #print("acc",acc)
                #print(tree.predict(pd.DataFrame(train)))
            accuracy_validation[depth] = avg_acc/4
        value = max(accuracy_validation, key = accuracy_validation.get)
        tree = DecisionTree(criterion="gini_index",max_depth=value)
        train = pd.DataFrame(train)
        train_label = pd.Series(train_label,dtype="category")

        tree.fit(train,train_label)
        #tree = tree_iris(train,value,0)
        print("Accuracy is,",accuracy(tree.predict(test),test_label), " for iteration",i+1, ". The depth of the optimal tree is ",value)

X, y = dataset.data, dataset.target


print("nested_cross_validation")
nested_cross_validation(X,y)





# X, y = dataset.data, dataset.target 
print("5 fold cross validation")
cross_validtion_5_fold(X,y,8)

#reference : cross validation function are studired from :
#https://github.com/sdeepaknarayanan/Machine-Learning/blob/master/Assignment%201/Decision%20Trees%20-%20Q2%2C%20Q3%2C%20Q4%2C%20Q5%2C%20Q6%2C%20Q7.ipynb