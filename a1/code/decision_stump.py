import numpy as np
import utils

def fit(X, y):
    # YOUR CODE HERE FOR Q2.1
    N,D = X.shape

    y_mode = utils.mode(y)
    splitSat = y_mode
    splitVariable = None
    splitValue = None
    splitNot = None

    minError = np.sum(y != y_mode)
    if np.unique(y).size > 1:
        for d in range(D):
            thresholds = np.unique(X[:,d])
            for threshold in thresholds[:-1]:
                y_above = utils.mode(y[X[:, d] > threshold])
                y_not = utils.mode(y[X[:, d] <= threshold])
                #fill in y_pred with value of highest frequency above threshold
                y_pred = y_above * np.ones(N)
                #change those not above threshold to the highest value appears
                #that is not above threshold
                y_pred[X[:, d] <= threshold] = y_not
                #compute errors
                errors = np.sum(y_pred != y)
                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    splitVariable = d
                    splitValue = threshold
                    splitSat = y_above
                    splitNot = y_not
    model=dict()
    model["splitVariable"] = splitVariable
    model["splitValue"] = splitValue
    model["splitSat"] = splitSat
    model["splitNot"] = splitNot
    model["predict"] = predict
    print(model)
    return model



def predict(model, X):
    """ YOUR CODE HERE FOR Q2.1 """
    """splitVariable is the feature we want to evaluate"""
    splitVariable = model["splitVariable"]
    """threshold"""
    splitValue = model["splitValue"]
    splitSat = model["splitSat"]
    splitNot = model["splitNot"]

    M, D = X.shape
    X = np.round(X)

    if splitVariable is None:
        return splitSat * np.ones(M)

    y_pred = np.zeros(M)

    for m in range(M):
        """use > here, only difference btw predict and predict_equality"""
        if X[m, splitVariable] > splitValue:
            y_pred[m] = splitSat
        else:
            y_pred[m] = splitNot

    return y_pred


#### DECISION STUMP EQUALITY
def fit_equality(X, y):
    N, D = X.shape


    y_mode = utils.mode(y)

    splitSat = y_mode
    splitVariable = None
    splitValue = None
    splitNot = None

    minError = np.sum(y != y_mode)

    # Check if labels are not all equal
    if np.unique(y).size > 1:
        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] == value])
                y_not = utils.mode(y[X[:,d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    splitVariable = d
                    splitValue = value
                    splitSat = y_sat
                    splitNot = y_not

    # Store variables as dict
    model = {}

    model["splitVariable"] = splitVariable
    model["splitValue"] = splitValue
    model["splitSat"] = splitSat
    model["splitNot"] = splitNot
    model["predict"] = predict_equality

    return model

def predict_equality(model, X):
    splitVariable = model["splitVariable"]
    splitValue = model["splitValue"]
    splitSat = model["splitSat"]
    splitNot = model["splitNot"]

    M, D = X.shape
    X = np.round(X)

    if splitVariable is None:
        return splitSat * np.ones(M)

    y_pred = np.zeros(M)

    for m in range(M):
        if X[m, splitVariable] == splitValue:
            y_pred[m] = splitSat
        else:
            y_pred[m] = splitNot

    return y_pred
