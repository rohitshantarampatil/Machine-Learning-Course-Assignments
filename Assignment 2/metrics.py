import math
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    if type(y_hat)!=list:
        assert(y_hat.size == y.size)
        y_hat = y_hat.tolist()
        y = y.tolist()
    # TODO: Write here
    count = 0
    for i in range(len(y)):
        if y_hat[i] == y[i]:
            count+=1
    return(count/len(y_hat))




def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """

    if type(y_hat)!=list:
        assert(y_hat.size == y.size)
        y_hat = y_hat.tolist()
        y = y.tolist()
    # TODO: Write here
    count = 0
    count2 =0
    for i in range(len(y)):
        if y_hat[i] == y[i] and y[i]==cls:
            count+=1
        if y_hat[i]==cls:
            count2+=1
    if count2==0:
        return 1
    return(count/count2)

    

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    if type(y_hat)!=list:
        assert(y_hat.size == y.size)
        y_hat = y_hat.tolist()
        y = y.tolist()
    # TODO: Write here
    count = 0
    count2 =0
    for i in range(len(y)):
        if y_hat[i] == y[i] and y[i]==cls:
            count+=1
        if y[i]==cls:
            count2+=1
        if count2==0:
            return 1
    return(count/count2)


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    y = y.tolist()
    y_hat = y_hat.tolist()
    
    sum = 0
    for i in range(len(y)):
        sum += (y_hat[i] - y[i])**2
    sum = sum/len(y)
    return math.sqrt(sum)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    y = y.tolist()
    y_hat = y_hat.tolist()
    
    sum = 0
    for i in range(len(y)):
        sum += abs(y_hat[i] - y[i])
    sum = sum/len(y)
    return sum
