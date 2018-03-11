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
def CalculateResults(X, Y, b, threshold):
    result_opt = Classify_Prediction(X, b, threshold)  # Make prediction
    return sum(result_opt == Y) / len(Y)  # Classify accuracy

def Classify_Prediction(X, b, threashold):
    return np.array(np.dot(X, b) > threashold)

def NormalizeData(data):
    # arr = [[255] * len(data[0])] * len(data)
    return data / 255  #image data 255 max.  Using min-max normailization

# Use equation from class, Don't re-invent wheel.  Code from Dr. Kang's Example:  http://ksuweb.kennesaw.edu/~mkang9/teaching/CS7267/code3/Linear_Regression.html
# def LinearRegression(X, y):
#     return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)
def binomial(x, n, p):
    return math.factorial(n) / (math.factorial(x) * math.factorial(n - x)) * (p ** x) * (1 - p) ** (n - x)

def Bernoulli(x, p):
    return (p ** x) * (1 - p) ** (1 - x)

def GradDecent(X, y, b):
    # E = math.exp(b[0] + np.dot(X, b))

    E = np.exp( np.dot(X., b))
    return - (E * np.dot(X, y) / (1 + E)) + np.dot(X, y)
    # sum = 0
    # for feature in X:
    #     sum += - (E *  feature  / ( 1 + E )) + (feature * y)

    # return math.factorial(n) / (math.factorial(x) * math.factorial(n - x)) * (p ** x) * (1 - p) ** (n - x)
    # return  -np.dot(X.transpose(), y) + np.dot(np.dot(X.transpose(), X) ,b)

def Cost(X, y, b):
    return np.sum(np.dot(X, b) - y)**2


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

    # print (train_labels)

    # Normalize
    train_data = NormalizeData(train_data)
    test_data = NormalizeData(test_data)

    # likelihoods = []
    # test_ps = np.arange(0, 11) / 10.0
    # for j in range(0, len(test_ps)):
    #     test_p = np.array(test_ps[j]) * len(train_data[0])
    #
    #     Bernoulli_for_Each_Event = [Bernoulli(train_data[i], test_p) for i in range(0, len(train_data))]
    #     likelihood = np.exp(sum(np.log(Bernoulli_for_Each_Event)))
    #     likelihoods.append(likelihood)
    #     print (likelihood)
    #
    # # show the likelihood score of each p
    # plt.plot(test_ps, likelihoods)
    # plt.show()
    #
    # # maximum likelihood of the function
    # print(np.argmax(likelihoods))
    # test_ps[np.argmax(likelihoods)]

    b_est = np.zeros(len(train_data[0]))  # Array of zeros for each feature
    learn_rate = 1e-4
    r = 100

    plot = []
    for i in range(0, r):
        # Iterativly Calculates b_est based on learning rate and gradient decent
        b_est = b_est - learn_rate * GradDecent(train_data, train_labels, b_est)

        # Calculate cost for graph
        cost = Cost(train_data, train_labels, b_est)
        plot.append(cost)

    accuracy_est = CalculateResults(test_data, test_labels, b_est, threshold)

    # Display Accuracy
    # bdiff = sum(abs(b_opt - b_est))
    # print("b_opt: ", b_opt)
    # print("b_est: ", b_est)
    # print("b_opt accuracy", accuracy_opt)
    # print("b_est accuracy", accuracy_est)
    # print("B Diff: ", bdiff)


