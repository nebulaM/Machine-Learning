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
    X = model['X']
    y = model['y']
    k = model['k']
    R, C = Xtest.shape

    D = utils.euclidean_dist_squared(X, Xtest)
    # axis=0 alongs colmun, axis=1 alongs row
    # sort along column, np.argsort returns index of each element from low to high
    D_sort = np.argsort(D, axis=0)
    # y_pred has the same size as samples in X_test, which is row number of X_test
    y_pred=np.zeros(R)

    if k==1:
        y_pred[:]=y[D_sort[0, :]]
    else:
        for r in range(R):
            list_min_dist=np.zeros(k)
            for i in range(k):
                # ith shortest distance element from X to X_test
                list_min_dist[i]=y[D_sort[i, r]]
            y_pred[r]=utils.mode(list_min_dist)

    return y_pred

def report(k, X, y, Xtest, ytest):
    y_pred = predict(fit(X, y, k), Xtest)
    print("Training error for k=%d is %f" %(k, utils.classification_error(y, \
    y_pred)))
    print("Test error for k=%d is %.3f" %(k, utils.classification_error(ytest, \
    y_pred)))
