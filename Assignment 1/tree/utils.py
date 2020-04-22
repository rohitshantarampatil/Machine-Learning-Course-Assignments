import math
import pandas as pd
import numpy as np
def entropy(Y):


    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    Y = Y.tolist()
    lst = list(set(Y))
    count=[]
    for i in lst:
        count.append(Y.count(i))
    entropy = 0
    for i in range(len(lst)):
        if count[i]!=0:
            entropy-=(count[i]/sum(count))*math.log((count[i]/sum(count)),2)
        else:
            entropy-=0
    return entropy


def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outputs:
    > Returns the gini index as a float
    """
    Y = Y.tolist()
    lst = list(set(Y))
    count=[]
    for i in lst:
        count.append(Y.count(i))
    gini_index = 1

    for i in range(len(lst)):
        gini_index-=(count[i]/sum(count))**2

    return gini_index

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    #print(attr)
    gain = entropy(Y)
    Y = Y.tolist()
    lst = list(set(attr.tolist()))
    attr = attr.tolist()
    #rint(type(Y))
    #print("lst",lst)
    for i in lst:
        tmp = []
        for j in range(len(attr)):
            if attr[j]==i:
                tmp.append(Y[j])
        tmp = pd.Series(tmp)
        gain -= len(tmp)*entropy(tmp)/len(Y)
    return gain


def information_gain_analogue(Y, attr):
    #For the case of Discrete input real output
    gain = Y.var()
    Y = Y.tolist()
    lst = list(set(attr.tolist()))
    attr = attr.tolist()
    #rint(type(Y))
    #print("lst",lst)
    for i in lst:
        tmp = []
        for j in range(len(attr)):
            if attr[j]==i:
                tmp.append(Y[j])
        tmp = pd.Series(tmp)
        gain -= len(tmp)*(tmp.var())/len(Y)
    return gain

def information_gain_RIDO(Y,attr):
    attr = attr.rename("attr")
    df = pd.concat([Y,attr],axis=1)
    #df["attr"] = attr
    df = df.sort_values(by = "attr")
    #print(df)

    attr = df.pop("attr")
    Y = df.pop("label")

    gain = entropy(Y)
    Y = Y.tolist()
    attr = attr.tolist()
    splitpoint = None
    for i in range(len(Y)-1):
        if Y[i]!=Y[i+1]:
            splitpoint = i
            break
    if splitpoint==None:
        return(gain)
    else:
        gain = gain - (len(Y)-(splitpoint+1))*entropy(pd.Series(Y[i+1:]))
        return gain

def min_var(X,columns):
    var = 45464646
    min_attr = None
    for attribute in columns:
        maindict_freq = dict()
        for j in X[attribute]:
            if j not in maindict_freq:
                maindict_freq[j]=1
            elif j in maindict_freq:
                maindict_freq[j]+=1
        for i in maindict_freq:
            maindict_freq[i] = maindict_freq[i]/len(X)
        unique_values = [k for k in maindict_freq]
        lst = []
        for j in unique_values:
            t = X[X[attribute]==j]
            tempval = np.std(t['y'])
            tempval = tempval*len(t)/len(X)
            lst.append(tempval)
        lst_sum = sum(lst)
        if lst_sum<var:
            var = lst_sum
            min_attr = attribute
    return min_attr
        

    
