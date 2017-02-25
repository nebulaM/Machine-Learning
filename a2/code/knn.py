"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

def fit(X, y, k):
    """
    Parameters
    ----------
    X : an N by D numpy array
    y : an N by 1 numpy array of integers in {1,2,3,...,c}
    k : the k in k-NN
    """
    # Just memorize the training dataset
    model = dict()
    model['X'] = X
    model['y'] = y
    model['k'] = k
    model['predict'] = predict
    return model

def predict(model, Xtest):
    """ YOUR CODE HERE """
    raise NotImplementedError
