import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures
# from sklearn.preprocessing import PolynomialFeatures


X = np.array([1,2])
poly = PolynomialFeatures(2)
print(poly.transform(X))
