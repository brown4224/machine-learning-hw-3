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

def CalculateResults(X, Y, weights, threshold):
    # result_opt = Classify_Prediction(X, b, threshold)  # Make prediction
    print (predict(X, weights) )
    result_opt = predict(X, weights) > threshold
    return sum(result_opt == Y) / len(Y)  # Classify accuracy

# def Classify_Prediction(X, w, threashold):
#     return np.array(np.dot(X, b) > threashold)

# Formula from class: http://ksuweb.kennesaw.edu/~mkang9/teaching/CS7267/code4/Maximum_Likelihood_Estimation.html
# def Gaussian_dist(x, m, std):
#     return np.exp(-(x-m)**2/(2*std**2))/((2 * math.pi * std**2)**0.5)

#def GradDecent(X, y, b):
#    return  -np.dot(X.transpose(), y) + np.dot(np.dot(X.transpose(), X) ,b)
# def GradDecent(X, y, b):
#                                         # np.dot(X.transpose(),  predictions - y)
#    return  -np.dot(X.transpose(), y) + np.dot(np.dot(X.transpose(), X) ,b)


#   Reference for formula: http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#gradient-descent
def sigmoid(z):
    return 1.0 /(1+ np.exp(-z))
def predict(X, weights):
    return sigmoid(np.dot(X, weights))
def GradDecent(X, y, weights):

    return np.dot( X.transpose(), predict(X, weights) - y ) / len(X)
    # print (grad)
    # return grad
    # return np.dot(X.transpose(),  predict(X, weights) - y)

def Cost(X, y, weights):
    val = sum((-y * np.log(predict(X, weights)) ) - ( 1 - y ) * np.log(1 - predict(X, weights)) )
    return val  / len(y)


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
    # weights = np.zeros(len(train_data[0]))  # Array of zeros for each feature
    weights =  np.ones(len(train_data[0]))  # Array of zeros for each feature
    learn_rate = 1e-4
    r = 1000

    plot = []
    for i in range(0, r):
        # Iterativly Calculates b_est based on learning rate and gradient decent
        weights = weights - learn_rate * GradDecent(train_data, train_labels, weights)

        # Calculate cost for graph
        cost = Cost(train_data, train_labels, weights)
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