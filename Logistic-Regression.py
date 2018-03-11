#  Environment: Centos 7, Python 3.6 with Pycharm
#  PIP requires numpy, sklearn, scipy
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# Websites used:
# Equations from here:  https://beckernick.github.io/logistic-regression-from-scratch/
# K Folding  http://scikit-learn.org/stable/modules/cross_validation.html#k-fold

# Other websites
# http://ksuweb.kennesaw.edu/~mkang9/teaching/CS7267/code4/Maximum_Likelihood_Estimation.html
# https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb
# http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#introduction

#####################################################################################################
###################################  Functions   ####################################################
#####################################################################################################
def NormalizeData(data):
    return data / 255.0
def NormalizeLabels(lables):
    arr = np.zeros(len(lables))
    m = np.amax(lables)
    for i in range(len(lables)):
        if lables[i] == m:
            arr[i] = 1
    return arr
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def prediction(X, weights):
    return np.dot(X, weights)
def likelihood(X, y, weights):
    return  np.sum( y * prediction(X, weights) - np.log( 1 + np.exp(prediction(X, weights))))
def GradDecent(X, y, weights):
    return np.dot( X.transpose(), y - sigmoid( prediction(X, weights)))

def CalculateResults(X, Y, weights, threshold):
    results = np.array(prediction(X, weights) > threshold)
    return sum(results == Y) / len(Y)  # Classify accuracy

def DisplayGraph(plotter):
    # Output Graph.  Flags for PyCharm
    plt.plot(plotter)
    plt.interactive(False)
    plt.show(block=True)


#####################################################################################################
#######################################  Main   #####################################################
#####################################################################################################
#  Main Start Program HERE
# Import data as Integers and remove labels

k_fold = 10
threshold = 0.5
data = np.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)

# Data Stats We Want
TPR = []  # True Positive Rates
FPR = []  # False Positive Rates

# Start K Folding
kf = KFold(n_splits=k_fold)
d = kf.split(data)
for train_index, test_index in d:
    # KFold returns array positions
    # Make new arrays to work with
    train_data = np.array(data[train_index])
    test_data = np.array(data[test_index])

    # Make sub arrays of labels and data
    train_labels = np.array(train_data[:,0])
    train_data = np.array(train_data[:, :-1])
    test_labels = np.array(test_data[:,0])
    test_data = np.array(test_data[:, :-1])


    # Normalize Data and Labels
    train_data = NormalizeData(train_data)
    test_data = NormalizeData(test_data)
    train_labels = NormalizeLabels(train_labels)
    test_labels = NormalizeLabels(test_labels)


    weights = np.zeros(len(train_data[0]))  # Array of zeros for each feature
    learn_rate = 1e-4
    r = 100

    plot = []
    for i in range(0, r):
        # Iterativly Calculates b_est based on learning rate and gradient decent
        weights = weights + learn_rate * GradDecent(train_data, train_labels, weights)


    accuracy = CalculateResults(test_data, test_labels, weights, threshold)
    print (accuracy)


    # DisplayGraph(plot)

