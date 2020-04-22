import math
import pandas as pd
from tree.utils import *
from metrics import *
import numpy as np
from collections import defaultdict
np.random.seed(42)

# y = [1,1,1,0,0]
# y = pd.Series(y)

# y_hat = [0,1,0,0,1]
# y_hat = pd.Series(y_hat)

#print(a)
###################################
# testing information gain
# attr = ["lol","dude","lol","dude","dude"]
# print(information_gain(a,attr))

####################################
#testing metric
# print(accuracy(y_hat,y))
# print(precision(y_hat,y,0))
# print(recall(y_hat,y,0))
# print(rmse(y_hat,y))
# print(mae(y_hat,y))

def get_attr(X):
	attr_lst = []
	for i in X:
		attr_lst.append(i)
	return(attr_lst)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")	
tree = defaultdict(dict)
attr_lst = get_attr(X)
X["label"] = y
data = X

# N = 30
# P = 5
# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randint(P, size = N) , dtype="category")
# tree = defaultdict(dict)
# attr_lst = get_attr(X)
# X["label"] = y
# data = X



# def get_attr_classes(X):
#     attr_dict=dict()
#     for i in X:
#     	attr_dict[i] =(X[i].unique())
#     return attr_dict

# attr_dict = get_attr_classes(X)




# def split_data_for_attr_class(X,y,attr,attr_class):
# 	X["label"] = y
# 	X = X.loc[X[attr]==attr_class]
# 	#print(X)
# 	y = X.pop("label")
# 	attr_class_column = X.pop(attr)
# 	return(X,y)
# def get_labels(y):
# 	return(y.unique())
def most_common_label(y):
	lst = y.tolist()
	labels = list(set(y))
	count=[]
	for i in labels:
		count.append(lst.count(i))
	return(labels[count.index(max(count))])


# def ID3(dictionary,data,target_attr,all_attr,attr_dict):
# 	X = data
# 	y = X.pop("label")
# 	# if len(y)==0 and len(X)==0:
# 	#print(labels)
# 	#if no target attr provided

# 	if (target_attr==None):
		
# 		inf_gain_lst = []
# 		for attr in all_attr:
# 			inf_gain_lst.append(information_gain(y,X[attr]))
# 		target_attr = all_attr[inf_gain_lst.index(max(inf_gain_lst))]
# 		dictionary["root"]=target_attr
# 	if len(all_attr)==0:
# 		dictionary[target_attr]["label"] = most_common_label(y)
# 		return

# 	if target_attr in all_attr:
# 		all_attr.remove(target_attr)
# 	attr_classes = attr_dict[target_attr]   	
	
# 	for attr_class in attr_classes:
# 		X_new,y_new= split_data_for_attr_class(X,y,target_attr,attr_class)
		
# 		if y_new.unique().size == 1:
# 			#print(y_new)
# 			dictionary[target_attr][attr_class] = " label " + str(y_new.iloc[0])
# 		elif(len(X_new)==0):
# 			dictionary[target_attr][attr_class] = " label " + str(most_common_label(y))
# 		else:
# 			inf_gain_lst_new=[]
# 			all_attr_new = get_attr(X_new)
# 			#print("all_attr_new",all_attr_new)
# 			for attribute in all_attr_new:
# 				#print(y_new,X_new[attribute])
# 				#print(information_gain(y_new,X_new[attribute]))
# 		def get_attr(X):
# 	attr_lst = []
# 	for i in X:
# 		attr_lst.append(i)
# 	return(attr_lst)
	

# N = 30
# P = 5
# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randint(P, size = N) , dtype="category")
# tree = defaultdict(dict)
# attr_lst = get_attr(X)
# X["label"] = y
# data = X



# def get_attr_classes(X):
#     attr_dict=dict()
#     for i in X:
#     	attr_dict[i] =(X[i].unique())
#     return attr_dict

# attr_dict = get_attr_classes(X)




# def split_data_for_attr_class(X,y,attr,attr_class):
# 	X["label"] = y
# 	X = X.loc[X[attr]==attr_class]
# 	#print(X)
# 	y = X.pop("label")
# 	attr_class_column = X.pop(attr)
# 	return(X,y)
# def get_labels(y):
# 	return(y.unique())
# def most_common_label(y):
# 	lst = y.tolist()
# 	labels = list(set(y))
# 	count=[]
# 	for i in labels:
# 		count.append(lst.count(i))
# 	return(labels[count.index(max(count))])


# def ID3(dictionary,data,target_attr,all_attr,attr_dict):
# 	X = data
# 	y = X.pop("label")
# 	# if len(y)==0 and len(X)==0:
# 	#print(labels)
# 	#if no target attr provided

# 	if (target_attr==None):
		
# 		inf_gain_lst = []
# 		for attr in all_attr:
# 			inf_gain_lst.append(information_gain(y,X[attr]))
# 		target_attr = all_attr[inf_gain_lst.index(max(inf_gain_lst))]
# 		dictionary["root"]=target_attr
# 	if len(all_attr)==0:
# 		dictionary[target_attr]["label"] = most_common_label(y)
# 		return

# 	if target_attr in all_attr:
# 		all_attr.remove(target_attr)
# 	attr_classes = attr_dict[target_attr]   	
	
# 	for attr_class in attr_classes:
# 		X_new,y_new= split_data_for_attr_class(X,y,target_attr,attr_class)
		
# 		if y_new.unique().size == 1:
# 			#print(y_new)
# 			dictionary[target_attr][attr_class] = " label " + str(y_new.iloc[0])
# 		elif(len(X_new)==0):
# 			dictionary[target_attr][attr_class] = " label " + str(most_common_label(y))
# 		else:
# 			inf_gain_lst_new=[]
# 			all_attr_new = get_attr(X_new)
# 			#print("all_attr_new",all_attr_new)
# 			for attribute in all_attr_new:
# 				#print(y_new,X_new[attribute])
# 				#print(information_gain(y_new,X_new[attribute]))
# 				inf_gain_lst_new.append(information_gain(y_new,X_new[attribute]))
# 			target_attr_new = all_attr_new[inf_gain_lst_new.index(max(inf_gain_lst_new))]
# 			#print("target_attr_new : ",target_attr_new)
# 			X_new["label"] = y_new
# 			data = X_new
# 			#print(dictionary)
# 			dictionary[target_attr][attr_class]=target_attr_new
# 			ID3(dictionary,data,target_attr_new,all_attr_new,attr_dict)



# ID3(tree,data,None,attr_lst,attr_dict)
# #print(tree)
# #print(information_gain(y,X[4]))
# #print(y)

# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# #print(X.dtypes[0])
# #print(X.iloc[3])

# def predict_single_example(ex,dictionary,node):
# 	# print("node ",node)
# 	#print(node)
# 	#print(ex)
# 	if type(node)==str:
# 		return(node)
# 	else:
# 		attr_type = ex.iloc[node]
# 		#print(dictionary[node][attr_type])
# 		return(predict_single_example(ex,dictionary,dictionary[node][attr_type]))


# #print(X)
# def predict_label(X,dictionary):
# 	y_hat_temp = []
# 	y_hat = []
# 	for i in range(len(X)):
# 		y_hat_temp.append(predict_single_example(X.iloc[i],tree,tree["root"])) 	
# 	for j in y_hat_temp:
# 		trash,label = j.split()
# 		y_hat.append(int(label))
# 	return(pd.Series(y_hat))

# y_hat = predict_label(X,tree)
# print(accuracy(y_hat,y))
# 		inf_gain_lst_new.append(information_gain(y_new,X_new[attribute]))
# 			target_attr_new = all_attr_new[inf_gain_lst_new.index(max(inf_gain_lst_new))]
# 			#print("target_attr_new : ",target_attr_new)
# 			X_new["label"] = y_new
# 			data = X_new
# 			#print(dictionary)
# 			dictionary[target_attr][attr_class]=target_attr_new
# 			ID3(dictionary,data,target_attr_new,all_attr_new,attr_dict)



# ID3(tree,data,None,attr_lst,attr_dict)
# #print(tree)
# #print(information_gain(y,X[4]))
# #print(y)

# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# #print(X.dtypes[0])
# #print(X.iloc[3])

# def predict_single_example(ex,dictionary,node):
# 	# print("node ",node)
# 	#print(node)
# 	#print(ex)
# 	if type(node)==str:
# 		return(node)
# 	else:
# 		attr_type = ex.iloc[node]
# 		#print(dictionary[node][attr_type])
# 		return(predict_single_example(ex,dictionary,dictionary[node][attr_type]))


# #print(X)
# def predict_label(X,dictionary):
# 	y_hat_temp = []
# 	y_hat = []
# 	for i in range(len(X)):
# 		y_hat_temp.append(predict_single_example(X.iloc[i],tree,tree["root"])) 	
# 	for j in y_hat_temp:
# 		trash,label = j.split()
# 		y_hat.append(int(label))
# 	return(pd.Series(y_hat))

# y_hat = predict_label(X,tree)
# print(accuracy(y_hat,y))



################################################  DIRO #############################################


# def get_attr(X):
# 	attr_lst = []
# 	for i in X:
# 		attr_lst.append(i)
# 	return(attr_lst)
	


# N = 30
# P = 5
# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# y = pd.Series(np.random.randn(N))
# tree = defaultdict(dict)
# attr_lst = get_attr(X)
# X["label"] = y
# data = X

# print(data)


# def get_attr_classes(X):
#     attr_dict=dict()
#     for i in X:
#     	attr_dict[i] =(X[i].unique())
#     return attr_dict

# attr_dict = get_attr_classes(X)




# def split_data_for_attr_class(X,y,attr,attr_class):
# 	X["label"] = y
# 	X = X.loc[X[attr]==attr_class]
# 	#print(X)
# 	y = X.pop("label")
# 	attr_class_column = X.pop(attr)
# 	return(X,y)

# def avg(y):
# 	lst = y.tolist()
# 	return(sum(lst)/len(lst))


# def ID3(dictionary,data,target_attr,all_attr,attr_dict):
# 	X = data
# 	y = X.pop("label")
# 	# if len(y)==0 and len(X)==0:
# 	#print(labels)
# 	#if no target attr provided

# 	if (target_attr==None):
		
# 		inf_gain_analogue_lst = []
# 		for attr in all_attr:
# 			inf_gain_analogue_lst.append(information_gain_analogue(y,X[attr]))
# 		target_attr = all_attr[inf_gain_analogue_lst.index(max(inf_gain_analogue_lst))]
# 		dictionary["root"]=target_attr
# 	if len(all_attr)==0:
# 		dictionary[target_attr]["label"] = avg(y)
# 		return

# 	if target_attr in all_attr:
# 		all_attr.remove(target_attr)
# 	attr_classes = attr_dict[target_attr]   	
	
# 	for attr_class in attr_classes:
# 		X_new,y_new= split_data_for_attr_class(X,y,target_attr,attr_class)
		
# 		if y_new.unique().size == 1:
# 			#print(y_new)
# 			dictionary[target_attr][attr_class] = " label " + str(y_new.iloc[0])
# 		elif(len(X_new)==0):
# 			dictionary[target_attr][attr_class] = " label " + str(avg(y))
# 		else:
# 			inf_gain_lst_new=[]
# 			all_attr_new = get_attr(X_new)
# 			#print("all_attr_new",all_attr_new)
# 			for attribute in all_attr_new:
# 				#print(y_new,X_new[attribute])
# 				#print(information_gain(y_new,X_new[attribute]))
# 				inf_gain_lst_new.append(information_gain_analogue(y_new,X_new[attribute]))
# 			target_attr_new = all_attr_new[inf_gain_lst_new.index(max(inf_gain_lst_new))]
# 			#print("target_attr_new : ",target_attr_new)
# 			X_new["label"] = y_new
# 			data = X_new
# 			#print(dictionary)
# 			dictionary[target_attr][attr_class]=target_attr_new
# 			ID3(dictionary,data,target_attr_new,all_attr_new,attr_dict)





# ID3(tree,data,None,attr_lst,attr_dict)
# print(tree)
# #print(information_gain(y,X[4]))
# #print(y)

# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# #print(X.dtypes[0])
# #print(X.iloc[3])

# def predict_single_example(ex,dictionary,node):
# 	# print("node ",node)
# 	#print(node)
# 	#print(ex)
# 	if "label" in str(node):
# 		return(node)
# 	else:
# 		attr_type = ex.iloc[node]
# 		#print(dictionary[node][attr_type])
# 		return(predict_single_example(ex,dictionary,dictionary[node][attr_type]))


# #print(X)
# def predict_label(X,dictionary):
# 	y_hat_temp = []
# 	y_hat = []
# 	for i in range(len(X)):
# 		y_hat_temp.append(predict_single_example(X.iloc[i],tree,tree["root"])) 	
# 	for j in y_hat_temp:
# 		trash,label = j.split()
# 		y_hat.append(float(label))
# 	return(pd.Series(y_hat))


# y_hat = predict_label(X,tree)
# print(rmse(y_hat,y))
# #print(accuracy(y_hat,y))
############################################### DIRO ###############################################

def getsplitpoint(Y,attr):

	attr = attr.rename("attr")
	df = pd.concat([Y,attr],axis=1)
	#df["attr"] = attr
	df = df.sort_values(by = "attr")
	#print(df)

	attr = df.pop("attr")
	Y = df.pop("label")
	Y = Y.tolist()
	attr = attr.tolist()
	splitpoint = None
	for i in range(len(Y)-1):
		if Y[i]!=Y[i+1]:
			splitpoint = i
			break
	return(splitpoint)

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.splitpoint = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        print(type(X))
        print(type(X))
        self.no_of_classes = len(set(y))
        self.no_of_attributes = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict_RIDO(self, X):
        return [self._predict(inputs) for inputs in X]

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

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.no_of_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            feature, split_value = self._best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] < split_value
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = feature
                node.splitpoint = split_value
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.splitpoint:
                node = node.left
            else:
                node = node.right
        return node.predicted_class 
############################################### RIDO #################################################

# def ID3_RIDO(dictionary,data,target_attr,all_attr):
# 	X = data
# 	y = X.pop("label")
# 	if (target_attr==None):		
# 		inf_gain_lst = []
# 		for attr in all_attr:
# 			inf_gain_lst.append(information_gain_RIDO(y,X[attr]))

# 		target_attr = all_attr[inf_gain_lst.index(max(inf_gain_lst))]
# 		dictionary["root"] = target_attr

# 	if len(y.unique())==1:
# 		dictionary[target_attr][">"] = " label "+str(y.unique()[0])
# 		return
		
	

# 	inf_gain_lst = []
# 	for attr in all_attr:
# 		inf_gain_lst.append(information_gain_RIDO(y,X[attr]))
# 	# print(X)
# 	# print(inf_gain_lst)
# 	target_attr_new = all_attr[inf_gain_lst.index(max(inf_gain_lst))]
# 	X["label"]=y
# 	X = X.sort_values(by=target_attr_new)
# 	y =X.pop("label")



# 	splitpoint = getsplitpoint(y,X[attr])
# 	print("splitpoint",splitpoint)
# 	print(target_attr_new)
# 	X.reset_index(drop=True,inplace= True)
# 	y.reset_index(drop=True,inplace=True)
# 	print(X)

# 	splitvalue = (X[target_attr_new][splitpoint]+X[target_attr_new][splitpoint+1])/2
# 	print("splitvalue ", splitvalue)
# 	dictionary[target_attr_new]["<"+str(splitvalue)] = "label" + str(y[splitpoint])

# 	X["label"]=y
# 	data = X.loc[X[target_attr_new]>splitvalue]
# 	#print(data)
# 	# dictionary[target_attr_new][">" + str(splitvalue)] = ID3_RIDO(dictionary,data,target_attr_new,all_attr)
# 	#dictionary[target_attr_new][">" + str(splitvalue)] = "lol"
# 	ID3_RIDO(dictionary,data,target_attr_new,all_attr)
# ID3_RIDO(tree,data,None,attr_lst)
# print(tree)



if __name__ == "__main__":
    import sys
    from sklearn.datasets import load_iris

    dataset = load_iris()
    X, y = dataset.data, dataset.target  # pylint: disable=no-member
 
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X[30:], y[30:])
    #print(clf.predict_RIDO([[0, 0, 5, 1.5]]))
    y = y[:30]
    y_hat = clf.predict_RIDO(X[:30])
    print(accuracy(pd.Series(y_hat),pd.Series(y)))
########################################################################################################