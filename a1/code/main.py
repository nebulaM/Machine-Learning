import sys
import argparse

import utils
import pylab as plt

import numpy as np
import naive_bayes
import decision_stump
import decision_tree
import mode_predictor

from sklearn.tree import DecisionTreeClassifier

from scipy import stats

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "1.2", "2.1", "2.2", "3.1", "4.3"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Q1.1 - This should print the answers to Q 1.1

        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")

        """ YOUR CODE HERE"""
        #print(X)
        # part 1: min, max, mean, median and mode
        print("min: %.2f max: %.2f mean: %.2f median: %.2f"
        %(np.min(X),np.max(X),np.mean(X),np.median(X)))
        print(stats.mode(X))
        # part 2: quantiles
        for i in range (0,5):
            if i==0:
                percent=10
            elif i==4:
                percent=90
            else:
                percent=int(0.25*i*100)
            print("%d percent quantile is %.2f" %(percent,np.percentile(X, percent)))
        # part 3: maxMean, minMean, maxVar, minVar
        """ Each column of X is data from a city in names
            therefore select X wrt column, axis=0"""
        X_mean=np.mean(X,axis=0)
        """ np.where(condition) returns index of the element(s)
            in array that satisfies the condition"""
        print("City w/ max mean is %s"
        %(names[np.where(X_mean == np.max(X_mean))]))

        print("City w/ min mean is %s"
        %(names[np.where(X_mean == np.min(X_mean))]))

        X_var=np.var(X,axis=0)
        print("City w/ max var is %s"
        %(names[np.where(X_var == np.max(X_var))]))
        print("City w/ min var is %s"
        %(names[np.where(X_var == np.min(X_var))]))
        # part 4: correlation between columns
        min_cor=[sys.float_info.max,0,0]
        max_cor=[sys.float_info.min,0,0]
        R=np.corrcoef(X.T)
        #print(R)
        #print(names)
        """shape returns [row,column]"""
        for i in range(0,R.shape[1]-2):
            for j in range(i+1,R.shape[1]-1):
                if min_cor[0]>R[i][j]:
                    min_cor[0]=R[i][j]
                    min_cor[1]=i
                    min_cor[2]=j
                if max_cor[0]<R[i][j]:
                    max_cor[0]=R[i][j]
                    max_cor[1]=i
                    max_cor[2]=j
        print("min correlation is %.4f btw cities %s and %s"
            %(min_cor[0],names[min_cor[1]],names[min_cor[2]]))
        print("max correlation is %.4f btw cities %s and %s"
        %(max_cor[0],names[max_cor[1]],names[max_cor[2]]))


    elif question == "1.2":
        # Q1.2 - This should plot the answers to Q 1.2
        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")
        plt.plot(X)
        #plt.axis(names)
        plt.show()
        # Plot required figures

        """ YOUR CODE HERE"""

    elif question == "2.1":
        # Q2.1 - Decision Stump with the inequality rule Implementation

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        model = mode_predictor.fit(X, y)
        y_pred = mode_predictor.predict(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump with equality rule
        model = decision_stump.fit_equality(X, y)
        y_pred = decision_stump.predict_equality(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Decision Stump with equality rule error: %.3f"
              % error)

        # 4. Evaluate decision stump with inequality rule

        """ YOUR CODE HERE"""

        # PLOT RESULT
        utils.plotClassifier(model, X, y)
        fname = "../figs/q2.1_decisionBoundary.pdf"
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # Q2.2 - Decision Tree with depth 2

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = decision_tree.fit(X, y, maxDepth=2)

        y_pred = decision_tree.predict(model, X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

        # 3. Evaluate decision tree that uses information gain
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    elif question == "3.1":
        # Q3.1 - Training and Testing Error Curves

        # 1. Load dataset
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)


    elif question == "4.3":
        # Q4.3 - Train Naive Bayes

        # 1. Load dataset
        dataset = utils.load_dataset("newsgroups")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        # 2. Evaluate the decision tree model with depth 20
        model = DecisionTreeClassifier(criterion='entropy', max_depth=20)
        model.fit(X, y)
        y_pred = model.predict(X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Decision Tree Validation error: %.3f" % v_error)

        # 3. Evaluate the Naive Bayes Model
        model = naive_bayes.fit_wrong(X, y)

        y_pred = naive_bayes.predict(model, X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes Validation error: %.3f" % v_error)
