import matplotlib.pyplot as plt
import numpy as np
from softsvm import *


def first_experiment(): # for question 2
    # create lists for the min/ max/ avg errors on training and tests
    min_errors_test, max_errors_test, avg_errors_test = [], [], []
    min_errors_train, max_errors_train, avg_errors_train = [], [], []

    for n in range(1, 11): # we run for different lambda sizes: 10^n for n in {1, ... , 10}

        errors_train, errors_test = [], []

        for i in range(10): # for each lambda we run the prediction 10 times
            # get a random 100 training examples from the training set
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:100]]
            _trainy = trainy[indices[:100]]

            # get w using our softsvm
            w = softsvm(10 ** n, _trainX, _trainy)

            # label the test using w
            predicty = np.sign(testX @ w)
            errors_test.append(np.mean(predicty != testy))

            # label the train using w
            predict_train = np.sign(_trainX @ w)
            errors_train.append(np.mean(predict_train != _trainy))

        # make errors numpy arrays
        errors_train = np.array(errors_train)
        errors_test = np.array(errors_test)

        # add to min/max/avg
        min_errors_test.append(min(errors_test))
        max_errors_test.append(max(errors_test))
        avg_errors_test.append(np.average(errors_test))
        min_errors_train.append(min(errors_train))
        max_errors_train.append(max(errors_train))
        avg_errors_train.append(np.average(errors_train))

    # make errors numpy arrays
    min_errors_test = np.array(min_errors_test)
    max_errors_test = np.array(max_errors_test)
    avg_errors_test = np.array(avg_errors_test)
    min_errors_train = np.array(min_errors_train)
    max_errors_train = np.array(max_errors_train)
    avg_errors_train = np.array(avg_errors_train)

    lambda_values = np.array([10 ** n for n in range(1, 11)]) # np.array of lambda values
    fig, ax = plt.subplots()
    ax.errorbar(lambda_values, avg_errors_test, color='black',
                yerr=[avg_errors_test - min_errors_test, max_errors_test - avg_errors_test], capsize=5,
                label='Test Error')
    ax.errorbar(lambda_values, avg_errors_train, color='blue',
                yerr=[avg_errors_train - min_errors_train, max_errors_train - avg_errors_train], capsize=5,
                label='Train Error')
    ax.set_xlabel('Lambda (log scale)')
    ax.set_ylabel('Error')
    ax.set_title('Training and Test Error with Different Lambdas')
    ax.set_xscale('log')
    ax.legend()
    plt.show()


def second_experiment(): # for question 2
    errors_train, errors_test = [], []
    lambda_values = [10 ** 1, 10 ** 3, 10 ** 5, 10 ** 8]

    for i in range(len(lambda_values)): # now we run each experiment with different lambda- once
        # Get a random 100 training examples from the training set
        indices = np.random.permutation(trainX.shape[0])
        _trainX = trainX[indices[:1000]]
        _trainy = trainy[indices[:1000]]

        # get w
        w = softsvm(lambda_values[i], _trainX, _trainy)

        # label the test using w
        predicty = np.sign(testX @ w)
        errors_test.append(np.mean(predicty != testy))

        # label the train using w
        predict_train = np.sign(_trainX @ w)
        errors_train.append(np.mean(predict_train != _trainy))

    errors_train = np.array(errors_train)
    errors_test = np.array(errors_test)
    lambda_values = np.array(lambda_values)

    fig, ax = plt.subplots()
    ax.scatter(lambda_values, errors_train, color='blue', label='Train Error')
    ax.scatter(lambda_values, errors_test, color='black', label='Test Error')
    ax.set_xlabel('Lambda (log scale)')
    ax.set_ylabel('Error')
    ax.set_title('Training and Test Error with Different Lambdas (Large Sample Size)')
    ax.set_xscale('log')
    ax.legend()
    plt.show()


data = np.load('EX2q2_mnist.npz')
trainX = data['Xtrain']
testX = data['Xtest']
trainy = data['Ytrain']
testy = data['Ytest']

first_experiment()
second_experiment()