import matplotlib.pyplot as plt
from softsvmpoly import *

data = np.load('EX3q2_data.npz')
trainX = data['Xtrain']
testX = data['Xtest']
trainy = data['Ytrain']
testy = data['Ytest']


def fold_cross_vali(l ,k, splitX, splitY): #2b
    errors = []
    for i in range(5):
        train_xi = np.vstack([splitX[j] for j in range(5) if j != i]) # get the sub-arrays that are not the i
        train_yi = np.hstack([splitY[j] for j in range(5) if j != i])
        fold_x = splitX[i] # get the ith subarray
        fold_y = splitY[i]
        w = softsvmpoly(l, k, train_xi, train_yi) # use the poly soft SVM of train_xi
        fold_k = (1 + np.dot(fold_x, train_xi.T)) ** k
        decision_values = np.dot(fold_k, w)
        predictions = np.sign(decision_values) # predictions of fold_x
        count = 0
        for j in range(len(predictions)):
            if predictions[j] != fold_y[j]:
                count += 1
        errors.append(count / len(predictions))
    return np.mean(errors) # return the average error for l and k


def cross_validation(): #2b
    lambdas = [1, 10, 100]
    ks = [2, 5, 8]
    split_x = np.asarray(np.split(trainX, 5)) # split trainX to 5 sub-arrays
    split_y = np.asarray(np.split(trainy, 5)) # split trainy to 5 sub-arrays

    errors = []
    for l in lambdas:
        for k in ks:
          error = fold_cross_vali(l, k, split_x, split_y)
          print(f"average error for l = {l}, k = {k} is: {error}")
          errors.append((l, k, error))

    best_pair = min(errors, key=lambda x: x[2]) # get the best pair by the lowest average error
    print(f"best pair: l = {best_pair[0]}, k = {best_pair[1]}")
    w = softsvmpoly(best_pair[0], best_pair[1], trainX, trainy) # train model on entire train samples

    fold_k = (1 + np.dot(testX, trainX.T)) ** best_pair[1]
    decision_values = np.dot(fold_k, w)
    predictions = np.sign(decision_values)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] != testy[i]:
            count += 1
    print(f"The average test error: {count/ len(predictions)}")


def run_and_plot(): # 2d
    l = 100
    ks = [3, 5, 8]
    (x0_min, x0_max) = (trainX[:, 0].min(), trainX[:, 0].max())
    (x1_min, x1_max) = (trainX[:, 1].min(), trainX[:, 1].max())
    for k in ks:
        w = softsvmpoly(l, k, trainX, trainy)
        grid = []
        for i in range(int(x0_min*100), int(x0_max*100)):
            row = []
            for j in range(int(x1_min*100), int(x1_max*100)):
                point = (j / 100., i / 100.)
                kernel = np.array([(1 + np.dot(xj, point)) ** k for xj in trainX])
                shaped_w = w.reshape(w.shape[0])
                prediction = np.sign(np.dot(shaped_w, kernel))
                if prediction == 1: # 1 is red
                    row.append([255, 0, 0])
                else: # -1 is blue
                    row.append([0, 0, 255])
            grid.insert(0, row)

        plt.imshow(grid, extent=[-1, 1, -1, 1])
        plt.title(f"for lambda=100 and k = {k}")
        plt.show()


cross_validation()
run_and_plot()
