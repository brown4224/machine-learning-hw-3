#  Environment: Centos 7, Python 3.6 with Pycharm
#  PIP requires numpy, sklearn, scipy
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
# http://scikit-learn.org/stable/modules/cross_validation.html#k-fold
#####################################################################################################
###################################  Functions   ####################################################
#####################################################################################################
def NormalizeData(data):
    return data / 255

def CalculateResults(X, Y, weights, b, threshold):
    prediction = predict(X, weights, b) > threshold
    return sum(prediction == Y) / len(Y)


    # Finish

    # result_opt = Classify_Prediction(X, b, threshold)  # Make prediction
    # print (predict(X, weights) )
    # result_opt = predict(X, weights) > threshold
    # return sum(result_opt == Y) / len(Y)  # Classify accuracy




#   Reference for formula: http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#gradient-descent
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    # return 1.0 /(1+ np.exp(-z))
def predict(X, weights, b):
    m = len(X[0])
    A = sigmoid(np.dot(weights.transpose(), X) + b)

    # return sigmoid(np.dot(X, weights))
def update_weights(X, y, weights, b):
    prediction = predict(X, weights, b)
    return 1.0 / len(X[0]) * np.dot(X, (prediction - y).transpose())

def update_bias(X, y, weights, b):
    prediction = predict(X, weights, b)
    return 1.0 / len(X[0]) * np.sum(prediction - y)


        # db = 1.0 / m* np.sum(A - Y)
    # return np.dot( X.transpose(), predict(X, weights) - y ) / len(X)


def Cost(X, y, weights, b):
    prediction = predict(X, weights, b)

    return np.squeeze( -1.0 / len(X[0]) * np.sum(y * np.log(prediction) + (1.0 - y) * np.log(1.0 - prediction))  )
    # val = sum((-y * np.log(predict(X, weights)) ) - ( 1 - y ) * np.log(1 - predict(X, weights)) )
    # return val  / len(y)


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

    print (train_labels)

    # Normalize
    train_data = NormalizeData(train_data)
    test_data = NormalizeData(test_data)


    # Grad Decent with Sigmoid gradiant
    weights = np.zeros(train_data.shape[0], 1)  # Array of zeros for each feature
    b=0
    # weights =  np.ones(len(train_data[0]))  # Array of ones for each feature
    learn_rate = 1e-4
    r = 1000

    plot = []
    for i in range(0, r):
        # Iterativly Calculates b_est based on learning rate and gradient decent
        # weights = weights - learn_rate * GradDecent(train_data, train_labels, weights)
        weights = weights - learn_rate * update_weights(train_data, train_labels, weights, b)
        b = b - learn_rate * update_bias(train_data, train_labels, weights, b)
        # Calculate cost for graph
        cost = Cost(train_data, train_labels, weights, b)
        plot.append(cost)

    accuracy = CalculateResults(test_data, test_labels, weights, threshold)
    print (accuracy)

    DisplayGraph(plot)

    # Logistic Regression
    # mean = np.mean(train_data, axis=0)  # axis 0: column,  axis 1: rows
    # std = np.std(train_data, axis=0)
    # # print (std)
    # print (len(mean))
    # std = np.array(2) * len(train_data[0])
    # likelihoods = []
    # r = np.arange(0, 11)
    # for j in range(0, len(r)):
    #     test_m = r[j]
    #     # mean = np.array(test_m)  * len(train_data[0])
    #     Gaussians = [Gaussian_dist(train_data[i], mean, std) for i in range(0, len(train_data))]
    #     # Gaussians = [Gaussian_dist(train_data[i], mean, std) for i in range(0, len(train_data))]
    #     likelihood = np.exp(sum(np.log(Gaussians)))
    #     likelihoods.append(likelihood)
    #     # print (likelihood)
    #
    # # maximum likelihood of the function
    # print (np.argmax(likelihoods))
    # test_ms[np.argmax(likelihoods)]

    # print (test_data[0])
    # # print (train)
    # print ("%s %s" % (len(train_data), len(test_data)))

# print(d)
