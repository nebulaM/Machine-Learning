import numpy as np

def fit(X, y):
    N, D = X.shape

    # Compute the number of class labels
    C = np.unique(y).size

    # Create a mapping from the labels to 0,1,2,...
    # so that we can store things in numpy arrays
    labels = dict()
    for index, label in enumerate(np.unique(y)):
        labels[index] = label

    # Compute the probability of each class i.e p(y==c)
    counts = np.zeros(C)

    for index, label in labels.items():
        counts[index] = np.sum(y==label)
    p_y = counts / N

    """ YOUR CODE HERE FOR Q4.3 """
    # init p(x|y)
    p_xy = np.zeros((D, C, 2))
    # Compute the conditional probabilities i.e.
    # p(x(i,j)=1 | y(i)==c) as p_xy
    # p(x(i,j)=0 | y(i)==c) as p_xy

    # after this loop, p_xy contains p(x n y)*N
    for n in range(N):
        for d in range(D):
            if X[n, d] == 1:
                for c in range(C):
                    if y[n]==labels[c]:
                        p_xy[d,c,1]+=1
            elif X[n, d] == 0:
                for c in range(C):
                    if y[n]==labels[c]:
                        p_xy[d,c,0]+=1
    # p(x n y)*N / N = p(x n y)
    # p(x n y)/p(y) = p(x|y)
    # recall p(y) =counts/N
    # only need to divide each column in p_xy by counts
    for d in range(D):
        p_xy[d,:,0]/=counts
        p_xy[d,:,1]/=counts
    # Save parameters in model as dict
    model = dict()

    model["p_y"] = p_y
    model["p_xy"] = p_xy
    model["n_classes"] = C
    model["labels"] = labels

    return model

def predict(model, X):
    # row, column == sample, feature
    N, D = X.shape
    # how many labels
    C = model["n_classes"]
    #p(x|y)
    p_xy = model["p_xy"]
    # probability of each feature
    p_y = model["p_y"]
    # label for each feature, in email-spam example, labels could be "spam" and "not spam"
    labels = model["labels"]

    y_pred = np.zeros(N)

    for n in range(N):
        # Compute the probability for each class
        # This could be vectorized but we've tried to provide
        # an intuitive version.
        probs = p_y.copy()

        for d in range(D):
            if X[n, d] == 1:
                for c in range(C):
                    # p(x n y) = p(y) * p(x|y)
                    # initially probs[c] = p(y)[c]
                    # after the * operation, probs[c] = p(x n y)[c]
                    probs[c] *= p_xy[d, c, 1]

            elif X[n, d] == 0:
                for c in range(C):
                    probs[c] *= p_xy[d, c, 0]

        # predict the label with highest prob
        y_pred[n] = labels[np.argmax(probs)]

    return y_pred
