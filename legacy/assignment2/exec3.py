from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import metrics
from sklearn.model_selection import train_test_split

from regression import normalEquation, logisticCost, linear, extendMatrix, logisticGradient
from statistics import normalizeFeature, sigmoid

style.use('fivethirtyeight')

df = pd.read_csv('datasets/breast_cancer.csv', header=None).sample(frac=1)
#
X = np.array(df.drop(columns=9, axis=1).T)
y = np.array(df.loc[:, 9])

# Remap the values to within the set [0,1].
y = [[0, 0, 0, 1, 1, 1][y_] for y_ in y]

# Normalized the values.
NormRandX = np.array(normalizeFeature(X))


def run(x_train, x_test, y_train, y_test):
	# Linear logistic function.
	Xe = extendMatrix(x_train, False, True)
	# Get the shape of the beta.
	linearBeta = normalEquation(Xe, y_train)

	alphaSet = [0.9, 0.5, 0.1, 0.05, 0.01, 0.005]
	alphaC = ['r', 'g', 'b', 'y', 'c', 'k']
	costs = []
	for i, alpha in enumerate(alphaSet):
		lineX = []
		lineY = []
		for n in range(1, 2000, 100):
			optimized = logisticGradient(Xe, linearBeta, y_train, alpha, n)
			combinationCost = logisticCost(Xe, optimized, y_train, linear)

			lineX.append(float(n))
			lineY.append(combinationCost)
			costs.append([combinationCost, optimized])
			print("alpha {}, iterations: {} J(B) : {}".format(alpha, n, round(combinationCost, 7)))

		plt.plot(lineX, lineY, color=alphaC[i], label="{}".format(alpha))
		plt.xlabel("iterations")
		plt.ylabel("cost")
		plt.legend(loc='upper left')
		plt.title("Cost in respect to number of iterations".format(alpha))
		plt.suptitle("breast_cancer.csv")
		plt.ylim(min(lineY) - 0.5, max(lineY) + 0.5)
	plt.show()

	# Determine the best theta.
	sortedCost = sorted(costs, key=itemgetter(0))
	cost, beta = sortedCost[0]
	print("The best beta {}, cost = {}".format(beta.tolist(), cost))

	threshold = 0.5
	accuracy = 0
	predictions = []
	for _x, _y in zip(x_test, y_test):
		p = sigmoid(linear(beta, _x))

		if p > threshold:
			p = 1
		else:
			p = 0
		predictions.append(p)
		if p == _y:
			accuracy = accuracy + 1

	cm = metrics.confusion_matrix(y_test, predictions)
	print(cm)
	# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
	print("Precision:", metrics.precision_score(y_test, predictions))
	print("Recall:", metrics.recall_score(y_test, predictions))

	# Compute the accuracy
	accuracyRate = accuracy / len(y_test)
	print("Accuracy: ", accuracyRate)


print("No Shuffle")
x_train, x_test, y_train, y_test = train_test_split(NormRandX.T, y, test_size=0.20, shuffle=False)
run(x_train.T, x_test, y_train, y_test)

print("Shuffle")
x_train, x_test, y_train, y_test = train_test_split(NormRandX.T, y, test_size=0.20, shuffle=True)
run(x_train.T, x_test, y_train, y_test)
