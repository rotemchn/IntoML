import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = trainX.shape[0]

    # the Gram matrix
    G = np.zeros((m, m))
    for row in range(m):
        for col in range(m):
            G[row, col] = (1 + np.dot(trainX[row], trainX[col])) ** k


    H = np.block([[G, np.zeros((m, m))], [np.zeros((m, m)), np.zeros((m, m))]])
    H *= 2 * l
    H += np.eye(2 * m)

    A = np.block([[np.zeros((m, m)), np.eye(m)], [np.diag(trainy) @ G.T, np.eye(m)]])

    u = np.hstack((np.full(m, float(0)), np.full(m, 1 / m)))

    v = np.hstack((np.zeros(m), np.ones(m)))

    sol = solvers.qp(matrix(H), matrix(u), -matrix(A), -matrix(v))
    alpha = np.array(sol["x"])[:m]
    return alpha


def simple_test():
    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
