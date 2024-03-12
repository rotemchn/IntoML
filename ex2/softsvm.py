import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m, d = trainX.shape # get the number of samples and the number of dimensions

    H = spmatrix(2 * l, range(d), range(d), (d + m, d + m))
    u = np.hstack((np.full(d, 0), np.full(m, 1 / m)))
    v = np.hstack((np.zeros(m), np.ones(m)))

    y_x_block = np.diag(trainy) @ trainX
    A = np.block([[np.zeros((m, d)), np.identity(m)], [y_x_block, np.identity(m)]])

    sol = solvers.qp(H, matrix(u), -matrix(A), -matrix(v))
    w = np.array(sol["x"])[:d]
    return w


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()



    # to run the experiments for question 2
    # first_experiment()
    # second_experiment()
