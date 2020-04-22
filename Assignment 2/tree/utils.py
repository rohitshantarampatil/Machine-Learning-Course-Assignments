
import math
import pandas as pd
def entropy(Y,weights):
    if type(Y) != list:
        Y = Y.tolist()    
    lst = list(set(Y))
    count=dict()
    for i in lst:
        count[i] = 0
    for i in lst:
        for j in range(len(Y)):
            if Y[j]==i:
                count[i]+=weights[j]
    entropy = 0
    for i in lst:
        if count[i]!=0:
            entropy-=(count[i]/sum(count.values()))*math.log((count[i]/sum(count.values())),2)
        else:
            entropy-=0
    return entropy

def information_gain(Y, attr,weights):
    gain = entropy(Y,weights)
    attr = attr.rename('attr')
    k = attr.to_frame()
    k['y'] = Y
    k['weights'] = weights
    k = k.sort_values(by = 'attr')
    k.reset_index(drop = True,inplace=True)
    Y = k.pop('y')
    weights = k.pop('weights')
    weights = weights.to_list()
    attr = k.pop('attr')
        
    splitpoints = []
    splitentropy = []
    for i in range(len(Y)-1):
        if Y[i]!=Y[i+1]:
            splitpoints.append(i)
            currentropy = gain
            currentropy-= entropy(Y[0:i+1],weights[0:i+1])
            currentropy-= entropy(Y[i+1:],weights[i+1:])
            splitentropy.append(currentropy)
    ind = splitpoints[splitentropy.index(max(splitentropy))]
    return max(splitentropy),(attr[ind]+attr[ind+1])/2,Y[ind],Y[ind+1]
    

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    pass

