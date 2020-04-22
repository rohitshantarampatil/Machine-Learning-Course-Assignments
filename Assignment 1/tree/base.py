"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *

np.random.seed(42)


######################  Necessary Functions Discret input Discrete Output ###########################
def get_attr_classes(X):
    attr_dict=dict()
    for i in X:
        attr_dict[i] =(X[i].unique())
    return attr_dict
def get_attr(X):
    attr = []
    for i in X:
        attr.append(i)
    return(attr)




def split_data_for_attr_class(X,y,attr,attr_class):
    X["label"] = y
    X = X.loc[X[attr]==attr_class]
    #print(X)
    y = X.pop("label")
    attr_class_column = X.pop(attr)
    return(X,y)
def get_labels(y):
    return(y.unique())
def most_common_label(y):
    lst = y.tolist()
    labels = list(set(y))
    count=[]
    for i in labels:
        count.append(lst.count(i))
    return(labels[count.index(max(count))])


def ID3(dictionary,data,target_attr,all_attr,attr_dict):
    X = data
    y = X.pop("label")
    labels = get_labels(y)

    #if no target attr provided

    if (target_attr==None):
        # print("yo bro")
        inf_gain_lst = []
        for attr in all_attr:
            inf_gain_lst.append(information_gain(y,X[attr]))
        target_attr = all_attr[inf_gain_lst.index(max(inf_gain_lst))]
        dictionary["root"]=target_attr
    if len(all_attr)==0:
        dictionary[target_attr]["label"] = most_common_label(y)
        return

    if target_attr in all_attr:
        all_attr.remove(target_attr)
    attr_classes = attr_dict[target_attr]       
    #print("target_attr : ",target_attr, "attr_classes : ", attr_classes)
    for attr_class in attr_classes:
        X_new,y_new= split_data_for_attr_class(X,y,target_attr,attr_class)
        
        if y_new.unique().size == 1:
            #print(y_new)
            dictionary[target_attr][attr_class] = " label " + str(y_new.iloc[0])
        elif(len(X_new)==0):
            dictionary[target_attr][attr_class] = " label " + str(most_common_label(y))
        else:
            inf_gain_lst_new=[]
            all_attr_new = get_attr(X_new)
            #print("all_attr_new",all_attr_new)
            for attribute in all_attr_new:
                #print(y_new,X_new[attribute])
                #print(information_gain(y_new,X_new[attribute]))
                inf_gain_lst_new.append(information_gain(y_new,X_new[attribute]))
            target_attr_new = all_attr_new[inf_gain_lst_new.index(max(inf_gain_lst_new))]
            # print("target_attr_new : ",target_attr_new)
            X_new["label"] = y_new
            data = X_new
            #print(dictionary)
            dictionary[target_attr][attr_class]=target_attr_new
            ID3(dictionary,data,target_attr_new,all_attr_new,attr_dict)



def predict_single_example(ex,dictionary,node):
    # print("node ",node)
    #print(node)
    #print(ex)
    if "label" in str(node):
        return(node)
    else:
        attr_type = ex[node]
        #print(dictionary[node][attr_type])
        return(predict_single_example(ex,dictionary,dictionary[node][attr_type]))


#print(X)
def predict_label(X,tree):
    #print(X)
    y_hat_temp = []
    y_hat = []
    for i in range(len(X)):
        y_hat_temp.append(predict_single_example(X.iloc[i],tree,tree["root"]))  
    for j in y_hat_temp:
        trash,label = j.split()
        
        if label.isdigit():
            y_hat.append(int(label))
        else:
            y_hat.append(label)
    return(pd.Series(y_hat))



#######################################################################
    
################################## Additional Functions : Discrete Input and Real Output ########################


def avg(y):
    lst = y.tolist()
    return(sum(lst)/len(lst))


def ID3_DIRO(dictionary,data,target_attr,all_attr,attr_dict):
    X = data
    y = X.pop("label")
    # if len(y)==0 and len(X)==0:
    #print(labels)
    #if no target attr provided

    if (target_attr==None):
        
        inf_gain_analogue_lst = []
        for attr in all_attr:
            inf_gain_analogue_lst.append(information_gain_analogue(y,X[attr]))
        target_attr = all_attr[inf_gain_analogue_lst.index(max(inf_gain_analogue_lst))]
        dictionary["root"]=target_attr
    if len(all_attr)==0:
        dictionary[target_attr]["label"] = avg(y)
        return

    if target_attr in all_attr:
        all_attr.remove(target_attr)
    attr_classes = attr_dict[target_attr]       
    
    for attr_class in attr_classes:
        X_new,y_new= split_data_for_attr_class(X,y,target_attr,attr_class)
        
        if y_new.unique().size == 1:
            #print(y_new)
            dictionary[target_attr][attr_class] = " label " + str(y_new.iloc[0])
        elif(len(X_new)==0):
            dictionary[target_attr][attr_class] = " label " + str(avg(y))
        else:
            inf_gain_lst_new=[]
            all_attr_new = get_attr(X_new)
            #print("all_attr_new",all_attr_new)
            for attribute in all_attr_new:
                #print(y_new,X_new[attribute])
                #print(information_gain(y_new,X_new[attribute]))
                inf_gain_lst_new.append(information_gain_analogue(y_new,X_new[attribute]))
            target_attr_new = all_attr_new[inf_gain_lst_new.index(max(inf_gain_lst_new))]
            #print("target_attr_new : ",target_attr_new)
            X_new["label"] = y_new
            data = X_new
            #print(dictionary)
            dictionary[target_attr][attr_class]=target_attr_new
            ID3_DIRO(dictionary,data,target_attr_new,all_attr_new,attr_dict)


def predict_single_example_DIRO(ex,dictionary,node):
                # print("node ",node)
                #print(node)
                #print(ex)
                if "label" in str(node):
                    return(node)
                else:
                    attr_type = ex.iloc[node]
                    #print(dictionary[node][attr_type])
                    return(predict_single_example_DIRO(ex,dictionary,dictionary[node][attr_type]))


            #print(X)
def predict_label_DIRO(X,tree):
    y_hat_temp = []
    y_hat = []
    for i in range(len(X)):
        y_hat_temp.append(predict_single_example_DIRO(X.iloc[i],tree,tree["root"]))  
    for j in y_hat_temp:
        trash,label = j.split()
        y_hat.append(float(label))
    return(pd.Series(y_hat))
##################################################################################################################
# REAL INPUT DISCRETE OUTPUT

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.splitpoint = 0
        self.left = None
        self.right = None

##################################################################################################################
        
def build_tree_regression(dataset,data,columns,min_data_length,parent):
    if len(data)<=min_data_length:
        return avg(data['y'])
    elif len(data)==0:
        return avg(dataset['y'])
    elif len(columns)==0:
        return parent
    else:
        parent = avg(data['y'])
        target_attribute = [min_var(data,columns)]
        tree = {target_attribute[0]:{}}
        columns.remove(target_attribute[0])
        target_data = []
        for i in data[target_attribute[0]]:
            if i not in target_data:
                target_data.append(i)
        temp_data = target_data
        for j in temp_data:
            j_data = data.loc[data[target_attribute[0]]==j]
            j_data = j_data.dropna()
            subtree = build_tree_regression(dataset,j_data,columns, min_data_length,parent)
            tree[target_attribute[0]][j] = subtree
    return tree

def predict_real_output(tree,instance):
    for index in tree.keys():
        if index in instance:
            try:
                potential = tree[index][instance[index]]
            except:
                #print("instance",instance)
                # return df['y'].mean()
                return 40.2
            potential =tree[index][instance[index]]  # Avoid referencement error
            if type(potential) == dict:
                return predict_real_output(potential, instance)
            else:
                return potential

###################################################################################################################
class DecisionTree():

    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.input_type = None
        self.output_type = None
        # self.no_of_classes = len(set(y))
        # self.no_of_attributes = X.shape[1]

    def fit(self, X, y):

        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        #############################################################################################
        #Discrete input and Discrete output
        self.input_type = X.dtypes[0].name
        self.output_type = y.dtype.name
        if (X.dtypes[0].name =="category" and y.dtype.name=="category"):
            maindict = defaultdict(dict)
            attr_lst = get_attr(X)
            attr_dict = get_attr_classes(X)
            X["label"] = y
            data = X
            ID3(maindict,data,None,attr_lst,attr_dict)
            self.tree = maindict

        elif (X.dtypes[0].name =="category" and y.dtype.name!="category"):
            maindict = defaultdict(dict)
            attr_lst = get_attr(X)
            attr_dict = get_attr_classes(X)
            X["label"] = y
            data = X
            ID3_DIRO(maindict,data,None,attr_lst,attr_dict)
            self.tree = maindict
        elif(X.dtypes[0].name !="category" and y.dtype.name=="category"):
            #print("yo")
            if type(X)!=np.ndarray:
                X = X.to_numpy()
            #print(type(X))
            if type(y)!=np.ndarray:
                y = y.to_numpy()
            #print(type(y))
            self.no_of_classes = len(set(y))
            self.no_of_attributes = X.shape[1]
            self.tree_ = self.progress_tree(X, y)
        else:
            attb= [i for i in X]
            X["y"] = y
            train = X
            tree_reg = build_tree_regression(train,train,attb,4,None)
            self.tree = tree_reg
  
###################################################### RIDO FUNCTIONS ####################################

    def predict_RIDO(self, X):
        return [self.temp_predict(inputs) for inputs in X]
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        #print("y")
        #print(y)
        
        # print("class_count",class_count)
        
        start_gini =gini_index(y)
        best_feature, best_split_threshold = None, None
        

        for feature in range(self.no_of_attributes):
            a = X[:, feature]
            temp1 = pd.Series(a)
            temp2 = pd.Series(y)
            a = temp1.rename("attr")
            y = temp2.rename("label")
            df = pd.concat([a,y],axis=1)
            df = df.sort_values(by="attr")
            a = df.pop("attr")
            y = df.pop("label")
            cutoff_values = a.tolist()

            classes = y.tolist()
            self.no_of_classes = max(list(set(classes)))+1
            #print(classes)
            #print("after")
            #print(cutoff_values,classes)


            labels_before = [0] * self.no_of_classes
            class_count = []
            set_y = list(set(y.tolist()))
            set_y.sort()
            for elem in range(self.no_of_classes):
                class_count.append(y.tolist().count(elem))

            num_right = class_count[:]
            for i in range(1, m):
                c = classes[i - 1]
                #print(self.no_of_classes)
                #print(c)
                labels_before[c] += 1
                num_right[c] -= 1
                gini_left = gini_index(pd.Series(labels_before))
                gini_right = gini_index(pd.Series(num_right))
                gini_index_temp = (i * gini_left + (m - i) * gini_right) / m
                if cutoff_values[i] == cutoff_values[i - 1]:
                    continue
                if gini_index_temp < start_gini:
                    start_gini = gini_index_temp
                    best_feature = feature
                    best_split_threshold = (cutoff_values[i] + cutoff_values[i - 1]) / 2
        return best_feature, best_split_threshold

    def progress_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.no_of_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            feature, split_value = self._best_split(X, y)
            if feature is not None:
                left_index = X[:, feature] < split_value
                X_l, y_l = X[left_index], y[left_index]
                X_r, y_r = X[~left_index], y[~left_index]
                node.feature_index = feature
                node.splitpoint = split_value
                node.left = self.progress_tree(X_l, y_l, depth + 1)
                node.right = self.progress_tree(X_r, y_r, depth + 1)
        return node

    def temp_predict(self, inputs):
        node = self.tree_
        #print(inputs)
        while node.left:
            if inputs[node.feature_index] < node.splitpoint:
                node = node.left
            else:
                node = node.right
        return node.predicted_class 

###############################################################################################
    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        if(self.input_type =="category" and self.output_type=="category"):
            y_hat = predict_label(X,self.tree)
            return y_hat

        elif(self.input_type =="category" and self.output_type!="category"):
            y_hat = predict_label_DIRO(X,self.tree)
            return y_hat
        elif(self.input_type !="category" and self.output_type=="category"):
            if type(X)!=np.ndarray:
                X = X.to_numpy()
            y_hat = [self.temp_predict(inputs) for inputs in X]
            return y_hat
        else:
            y_hat = np.zeros(len(X))
            for i in range(len(X)):
                val = predict_real_output(self.tree,X.iloc[i])
                y_hat[i] = val
            return y_hat

    def plot(self):
        pass
        """
        Function to plot the tree
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if(self.input_type=="category"):
            maindict = self.tree
            level = 0
            def printtree(maindict,node,level):
                if type(node)==str:
                    print(level*"  "+ node)
                    level-=1
                elif type(node)==int:
                    print(level*"  "+ str(node))
                    level+=1
                    printtree(maindict,maindict[node],level)
                elif type(node)==dict:
                    for i in node:
                        print(level*"  "+str(i))
                        # level+=1
                        printtree(maindict,node[i],level)
                else:
                    return 

            printtree(self.tree,maindict["root"],level)

#References: #https://github.com/sdeepaknarayanan/Machine-Learning/blob/master/Assignment%201/Decision%20Trees%20-%20Q2%2C%20Q3%2C%20Q4%2C%20Q5%2C%20Q6%2C%20Q7.ipynb