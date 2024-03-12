import numpy as np
import matplotlib.pyplot as plt
from nearest_neighbour import gensmallm
from nearest_neighbour import learnknn
from nearest_neighbour import predictknn

data = np.load('mnist_all.npz')

labels = [2, 3, 5, 6]
train_samples = [data['train2'], data['train3'], data['train5'], data['train6']]
test_samples = [data['test2'], data['test3'], data['test5'], data['test6']]
num_of_test_samples = len(data['test2']) + len(data['test3']) + len(data['test5']) + len(data['test6'])


def get_errors(m, k):
    errors = []
    x_test, y_test = gensmallm(test_samples, labels, num_of_test_samples)
    for i in range(10):
        x_train, y_train = gensmallm(train_samples, labels, m)

        classifier = learnknn(k, x_train, y_train)
        preds = predictknn(classifier, x_test)
        errors.append(np.mean(np.vstack(y_test) != np.vstack(preds)))
    
    return errors


def plot_k_1():
    samples = range(10, 101, 10)
    min_errors, max_errors, avg_errors = [], [], []
    for m in samples:
        errors = get_errors(m, 1)
        min_errors.append(min(errors))
        max_errors.append(max(errors))
        avg_errors.append(np.average(errors))

    fig, ax = plt.subplots()
    error_bars = [np.array(min_errors), np.array(max_errors)]
    ax.errorbar(samples, avg_errors, color='black', yerr=error_bars, capsize=5)
    ax.set_xlabel('sample size')
    ax.set_ylabel('average error rate')
    ax.set_title('k=1 min, max and average values of error as a function of the sample size m')
    plt.plot(samples, np.array(avg_errors), color='red', linewidth=2, marker='o', linestyle='-', markersize=8)
    plt.legend(["Average Error", "Min/Max Error"])
    plt.show()


def plot_k_1_to_12():
    errors = {}
    for k in range(1, 12):
        errors[k] = np.mean(get_errors(200, k))

    plt.plot(list(errors.keys()), list(errors.values()), marker='o')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.title(f'Error rates for different Ks')
    plt.show()


plot_k_1() # ex 2a-d
plot_k_1_to_12()  # ex 2e